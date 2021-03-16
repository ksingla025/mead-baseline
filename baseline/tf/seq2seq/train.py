import os
import time
import numpy as np
import tensorflow as tf
from eight_mile.tf.layers import create_session, reload_checkpoint
from eight_mile.tf.optz import EagerOptimizer
from eight_mile.utils import listify
from eight_mile.bleu import bleu

from baseline.utils import (
    convert_seq2seq_golds,
    convert_seq2seq_preds,
    get_model_file,
    get_metric_cmp
)

from baseline.train import Trainer, register_trainer, register_training_func
from baseline.tf.tfy import TRAIN_FLAG, SET_TRAIN_FLAG, get_shape_as_list, autograph_options, create_distribute_strategy

from baseline.model import create_model_for

# Number of batches to prefetch if using tf.datasets
NUM_PREFETCH = 2
# The shuffle buffer
SHUF_BUF_SZ = 5000


def loss(model, features, labels):
    # Claims its T, B, H
    logits = tf.transpose(model(features), [1, 0, 2])
    # So ok, then transpose this too
    labels = tf.transpose(labels, [1, 0])
    # TxB loss mask
    label_lengths = features['tgt_len']
    mx_seq_len = tf.reduce_max(label_lengths)-1
    labels = labels[1:mx_seq_len + 1, :]
    logits = logits[:mx_seq_len, :, :]
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    loss_mask = tf.cast(tf.sequence_mask(label_lengths-1), dtype=tf.float32)
    losses = losses * tf.transpose(loss_mask, [1, 0])

    losses = tf.reduce_sum(losses)
    losses /= tf.cast(tf.reduce_sum(label_lengths), tf.float32)
    return losses



def to_tensors(ts, src_lengths_key, dst=False):
    """Convert a data feed into a tuple of `features` (`dict`) and `y` values

    This method is required to produce `tf.dataset`s from the input data feed.
    Any fields ending with `_lengths` are ignored, unless they match the
    `src_lengths_key` or `tgt_lengths_key`, in which case, they are converted to `src_len` and `tgt_len`

    :param ts: The data feed to convert
    :param lengths_key: This is a field passed from the model params specifying source of truth of the temporal lengths
    :param dst: `bool` that says if we should prepare a `dst` tensor.  This is needed in distributed mode
    :return: A `tuple` of `features` and `y` (labels)
    """
    keys = ts[0].keys()
    # This is kind of a hack
    keys = [k for k in keys if '_lengths' not in k and k != 'ids'] + [src_lengths_key, "tgt_lengths"]

    features = dict((k, []) for k in keys)
    for sample in ts:
        for k in keys:
            for s in sample[k]:
                features[k].append(s)
    features['src_len'] = features[src_lengths_key]
    del features[src_lengths_key]
    features['tgt_len'] = features['tgt_lengths']
    del features['tgt_lengths']
    features = dict((k, np.stack(v).astype(np.int32)) for k, v in features.items())
    if dst:
        features['dst'] = features['tgt'][:, :-1]
    tgt = features.pop('tgt')

    return features, tgt


class Seq2SeqLoss(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.label_smoothing = kwargs.get("label_smoothing")
        if self.label_smoothing is not None:
            self.loss_fn = self.smoothed_loss
        else:
            self.loss_fn = self.loss

    def smoothed_loss(self, logits, labels):
        V = get_shape_as_list(logits)[-1]
        one_hot = tf.one_hot(labels, V)
        return tf.keras.losses.categorical_crossentropy(
            y_true=one_hot, y_pred=logits, from_logits=True, label_smoothing=self.label_smoothing
        )

    def loss(self, logits, labels):
        return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)

    def call(self, model, features, labels):
        # Claims its T, B, H
        logits = tf.transpose(model(features), [1, 0, 2])
        # So ok, then transpose this too
        labels = tf.transpose(labels, [1, 0])
        # TxB loss mask
        label_lengths = features['tgt_len']
        mx_seq_len = tf.reduce_max(label_lengths)-1
        labels = labels[1:mx_seq_len + 1, :]
        logits = logits[:mx_seq_len, :, :]
        losses = self.loss_fn(logits=logits, labels=labels)
        loss_mask = tf.cast(tf.sequence_mask(label_lengths-1), dtype=tf.float32)
        losses = losses * tf.transpose(loss_mask, [1, 0])

        losses = tf.reduce_sum(losses)
        losses /= tf.cast(tf.reduce_sum(label_lengths), tf.float32)
        return losses


class Seq2SeqTrainerEagerTf(Trainer):
    """Eager mode trainer for seq2sew

    The trainer can run in 2 modes: `dataset` and `feed_dict`.  When the former, the graph is assumed to
    be connected by features attached to the input so the `feed_dict` will only be used to pass dropout information.

    When the latter, we will use the baseline DataFeed to read the object into the `feed_dict`
    """
    def __init__(self, model_params, **kwargs):
        super().__init__()

        if type(model_params) is dict:
            self.model = create_model_for('seq2seq', **model_params)
        else:
            self.model = model_params

        self.tgt_rlut = kwargs['tgt_rlut']
        self.loss = Seq2SeqLoss(**kwargs)
        self.optimizer = EagerOptimizer(self.loss, **kwargs)
        self.nsteps = kwargs.get('nsteps', 500)
        self._checkpoint = tf.train.Checkpoint(optimizer=self.optimizer.optimizer, model=self.model)
        checkpoint_dir = '{}-{}'.format("./tf-seq2seq", os.getpid())
        self.bleu_n_grams = int(kwargs.get("bleu_n_grams", 4))

        self.checkpoint_manager = tf.train.CheckpointManager(self._checkpoint,
                                                             directory=checkpoint_dir,
                                                             max_to_keep=5)

    def checkpoint(self):
        """This method saves a checkpoint

        :return: None
        """
        self.checkpoint_manager.save()

    def recover_last_checkpoint(self):
        """Recover the last saved checkpoint

        :return: None
        """
        print(self._checkpoint.restore(self.checkpoint_manager.latest_checkpoint))

    @staticmethod
    def _num_toks(y):
        return tf.prod(get_shape_as_list(y))

    def _num_toks(self, lens):
        return tf.reduce_sum(lens)

    def train(self, ts, reporting_fns, dataset=True):
        """Train by looping over the steps

        For a `tf.dataset`-backed `fit_func`, we are using the previously wired `dataset`s
        in the model (and `dataset` is `True`).  For `feed_dict`, we convert the ts samples
        to `feed_dict`s and hand them in one-by-one

        :param ts: The training set
        :param reporting_fns: A list of reporting hooks
        :param dataset: (`bool`) Are we using `tf.dataset`s
        :return: Metrics
        """
        SET_TRAIN_FLAG(True)
        epoch_loss = tf.Variable(0.0)
        epoch_div = tf.Variable(0, dtype=tf.int32)
        nstep_loss = tf.Variable(0.0)
        nstep_div = tf.Variable(0, dtype=tf.int32)
        self.nstep_start = time.time()
        start = time.time()

        @tf.function
        def _train_step(features, y):
            """Replicated training step."""

            loss = self.optimizer.update(self.model, features, y)
            toks = self._num_toks(features['tgt_len'])
            report_loss = loss * tf.cast(toks, tf.float32)
            return report_loss, toks

        with autograph_options({"function_optimization": False, "layout_optimizer": False}):
            for features, y in ts:
                features['dst'] = y[:, :-1]
                step_report_loss, step_toks = _train_step(features, y)
                epoch_loss.assign_add(step_report_loss)
                nstep_loss.assign_add(step_report_loss)
                epoch_div.assign_add(step_toks)
                nstep_div.assign_add(step_toks)

                step = self.optimizer.global_step.numpy() + 1
                if step % self.nsteps == 0:
                    metrics = self.calc_metrics(nstep_loss.numpy(), nstep_div.numpy())
                    self.report(
                        step, metrics, self.nstep_start,
                        'Train', 'STEP', reporting_fns, self.nsteps
                    )
                    nstep_loss.assign(0.0)
                    nstep_div.assign(0)
                    self.nstep_start = time.time()

        epoch_loss = epoch_loss.numpy()
        epoch_div = epoch_div.numpy()
        metrics = self.calc_metrics(epoch_loss, epoch_div)
        self.train_epochs += 1
        self.report(
            self.train_epochs, metrics, start,
            'Train', 'EPOCH', reporting_fns
        )
        return metrics

    def calc_metrics(self, agg, norm):
        metrics = super().calc_metrics(agg, norm)
        metrics['perplexity'] = np.exp(metrics['avg_loss'])
        return metrics


    def _evaluate(self, es, reporting_fns, **kwargs):
        """Run the model with beam search and report Bleu.

        :param es: `tf.dataset` of input
        :param reporting_fns: Input hooks
        """
        preds = []
        golds = []
        start = time.time()

        for features, tgt in es:
            features['dst'] = tgt[:, :-1]
            tgt_lens = features.pop('tgt_len')
            top_preds = self.model.predict(features, make_input=False, **kwargs)[0]
            preds.extend(convert_seq2seq_preds(top_preds[:, 0, :], self.tgt_rlut))
            golds.extend(convert_seq2seq_golds(tgt, tgt_lens, self.tgt_rlut))
        metrics = {'bleu': bleu(preds, golds, self.bleu_n_grams)[0]}
        self.report(
            0, metrics, start, 'Test', 'EPOCH', reporting_fns
        )
        return metrics

    def test(self, vs, reporting_fns, phase='Valid', dataset=True, **kwargs):
        """Run an epoch of testing over the dataset

        If we are using a `tf.dataset`-based `fit_func`, we will just
        cycle the number of steps and let the `dataset` yield new batches.

        If we are using `feed_dict`s, we convert each batch from the `DataFeed`
        and pass that into TF as the `feed_dict`

        :param vs: A validation set
        :param reporting_fns: Reporting hooks
        :param phase: The phase of evaluation (`Test`, `Valid`)
        :param dataset: (`bool`) Are we using `tf.dataset`s
        :return: Metrics
        """
        SET_TRAIN_FLAG(False)
        if phase == 'Test':
            return self._evaluate(vs, reporting_fns, **kwargs)

        self.valid_epochs += 1

        total_loss = 0
        total_toks = 0
        preds = []
        golds = []

        start = time.time()
        for features, tgt in vs:
            features['dst'] = tgt[:, :-1]
            top_preds = self.model.predict(features, beam=1, make_input=False)[0]
            loss_value = self.loss(self.model, features, tgt).numpy()
            toks = tf.cast(self._num_toks(features['tgt_len']), tf.float32).numpy()
            total_loss += loss_value * toks
            total_toks += toks
            preds.extend(convert_seq2seq_preds(top_preds[:, 0, :], self.tgt_rlut))
            golds.extend(convert_seq2seq_golds(tgt, features['tgt_len'], self.tgt_rlut))

        metrics = self.calc_metrics(total_loss, total_toks)
        metrics['bleu'] = bleu(preds, golds, self.bleu_n_grams)[0]
        self.report(
            self.valid_epochs, metrics, start,
            phase, 'EPOCH', reporting_fns
        )
        return metrics


@register_training_func('seq2seq')
def fit_eager(model_params, ts, vs, es=None, **kwargs):
    """
    Train an language model using TensorFlow with `tf.dataset`.  This
    is the default behavior for training.

    :param model_params: The model (or parameters to create the model) to train
    :param ts: A training data set
    :param vs: A validation data set
    :param es: A test data set, can be None
    :param kwargs:
        See below

    :Keyword Arguments:
        * *do_early_stopping* (``bool``) --
          Stop after evaluation data is no longer improving.  Defaults to True
        * *verbose* (`dict`) A dictionary containing `console` boolean and `file` name if on
        * *epochs* (``int``) -- how many epochs.  Default to 20
        * *outfile* -- Model output file, defaults to classifier-model.pyth
        * *patience* --
           How many epochs where evaluation is no longer improving before we give up
        * *reporting* --
           Callbacks which may be used on reporting updates
        * *nsteps* (`int`) -- If we should report every n-steps, this should be passed
        * *ema_decay* (`float`) -- If we are doing an exponential moving average, what decay to us4e
        * *clip* (`int`) -- If we are doing gradient clipping, what value to use
        * *optim* (`str`) -- The name of the optimizer we are using
        * *lr* (`float`) -- The learning rate we are using
        * *mom* (`float`) -- If we are using SGD, what value to use for momentum
        * *beta1* (`float`) -- Adam-specific hyper-param, defaults to `0.9`
        * *beta2* (`float`) -- Adam-specific hyper-param, defaults to `0.999`
        * *epsilon* (`float`) -- Adam-specific hyper-param, defaults to `1e-8

    :return: None
    """

    epochs = int(kwargs.get('epochs', 5))
    patience = int(kwargs.get('patience', epochs))

    model_file = get_model_file('seq2seq', 'tf', kwargs.get('basedir'))

    do_early_stopping = bool(kwargs.get('do_early_stopping', True))

    best_metric = 0
    if do_early_stopping:
        early_stopping_metric = kwargs.get('early_stopping_metric', 'perplexity')
        early_stopping_cmp, best_metric = get_metric_cmp(early_stopping_metric, kwargs.get('early_stopping_cmp'))
        patience = kwargs.get('patience', epochs)
        print('Doing early stopping on [%s] with patience [%d]' % (early_stopping_metric, patience))

    reporting_fns = listify(kwargs.get('reporting', []))
    print('reporting', reporting_fns)

    batchsz = kwargs['batchsz']
    test_batchsz = kwargs.get('test_batchsz', batchsz)
    tgt_key = model_params.get('tgt_key')

    src_lengths_key = model_params.get('src_lengths_key')
    train_dataset = tf.data.Dataset.from_tensor_slices(to_tensors(ts, src_lengths_key))
    train_dataset = train_dataset.shuffle(buffer_size=SHUF_BUF_SZ)
    train_dataset = train_dataset.batch(batchsz, drop_remainder=False)
    train_dataset = train_dataset.prefetch(NUM_PREFETCH)

    valid_dataset = tf.data.Dataset.from_tensor_slices(to_tensors(vs, src_lengths_key))
    valid_dataset = valid_dataset.batch(batchsz, drop_remainder=False)
    valid_dataset = valid_dataset.prefetch(NUM_PREFETCH)

    trainer = Seq2SeqTrainerEagerTf(model_params, **kwargs)
    last_improved = 0
    SET_TRAIN_FLAG(True)

    for epoch in range(epochs):

        trainer.train(train_dataset, reporting_fns)
        test_metrics = trainer.test(valid_dataset, reporting_fns, phase='Valid')

        if do_early_stopping is False:
            trainer.checkpoint()
            trainer.model.save(model_file)

        elif early_stopping_cmp(test_metrics[early_stopping_metric], best_metric):
            last_improved = epoch
            best_metric = test_metrics[early_stopping_metric]
            print('New best %.3f' % best_metric)
            trainer.checkpoint()
            trainer.model.save(model_file)

        elif (epoch - last_improved) > patience:
            print('Stopping due to persistent failures to improve')
            break

    if do_early_stopping is True:
        print('Best performance on %s: %.3f at epoch %d' % (early_stopping_metric, best_metric, last_improved))

    if es is not None:
        print('Reloading best checkpoint')
        trainer.recover_last_checkpoint()
        test_dataset = tf.data.Dataset.from_tensor_slices(to_tensors(es, src_lengths_key))
        test_dataset = test_dataset.batch(test_batchsz, drop_remainder=False)
        test_dataset = test_dataset.prefetch(NUM_PREFETCH)
        trainer.test(test_dataset, reporting_fns, phase='Test')


class Seq2SeqTrainerDistributedTf(Trainer):
    """A Trainer to use for eager distributed mode
    """

    def __init__(self, model_params, **kwargs):
        super().__init__()
        if type(model_params) is dict:
            self.model = create_model_for('seq2seq', **model_params)
        else:
            self.model = model_params

        self.tgt_rlut = kwargs['tgt_rlut']
        self.optimizer = EagerOptimizer(loss, **kwargs)
        self.nsteps = kwargs.get('nsteps', 500)
        self._checkpoint = tf.train.Checkpoint(optimizer=self.optimizer.optimizer, model=self.model)
        checkpoint_dir = '{}-{}'.format("./tf-seq2seq", os.getpid())

        self.checkpoint_manager = tf.train.CheckpointManager(self._checkpoint,
                                                             directory=checkpoint_dir,
                                                             max_to_keep=5)
        strategy_type = kwargs.get('strategy_type', 'mirror')
        gpus = int(kwargs.get('gpus', 1))
        endpoint = kwargs.get('endpoint')
        self.strategy = create_distribute_strategy(strategy_type, endpoint, gpus)
        self.bleu_n_grams = int(kwargs.get("bleu_n_grams", 4))

    def checkpoint(self):
        """This method saves a checkpoint

        :return: None
        """
        self.checkpoint_manager.save()

    def recover_last_checkpoint(self):
        """Recover the last saved checkpoint

        :return: None
        """
        print(self._checkpoint.restore(self.checkpoint_manager.latest_checkpoint))

    @staticmethod
    def _num_toks(y):
        return tf.prod(get_shape_as_list(y))

    def _num_toks(self, lens):
        return tf.reduce_sum(lens)

    def train(self, ts, reporting_fns, steps=0):
        """Train by looping over the steps

        For a `tf.dataset`-backed `fit_func`, we are using the previously wired `dataset`s
        in the model (and `dataset` is `True`).  For `feed_dict`, we convert the ts samples
        to `feed_dict`s and hand them in one-by-one

        :param ts: The training set
        :param reporting_fns: A list of reporting hooks
        :param dataset: (`bool`) Are we using `tf.dataset`s
        :return: Metrics
        """
        strategy = self.strategy

        # num_replicas = strategy.num_replicas_in_sync
        def _replicated_train_step(inputs):
            features, y = inputs
            per_replica_loss = self.optimizer.update(self.model, features, y)
            per_replica_toks = self._num_toks(features['tgt_len'])
            per_replica_report_loss = per_replica_loss * tf.cast(per_replica_toks, tf.float32)
            return per_replica_report_loss, per_replica_toks

        with strategy.scope():
            SET_TRAIN_FLAG(True)
            epoch_loss = tf.Variable(0.0)
            epoch_div = tf.Variable(0, dtype=tf.int32)
            nstep_loss = tf.Variable(0.0)
            nstep_div = tf.Variable(0, dtype=tf.int32)
            self.nstep_start = time.time()
            start = time.time()

            @tf.function
            def _distributed_train_step(inputs):
                per_replica_loss, per_replica_toks = strategy.run(_replicated_train_step, args=(inputs,))
                total_step_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)
                total_toks = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_toks, axis=None)
                return total_step_loss, total_toks

            with autograph_options({"function_optimization": False, "layout_optimizer": False}):
                train_iter = iter(ts)
                for i in range(steps):
                    features, y = next(train_iter)
                    step_report_loss, step_toks = _distributed_train_step((features, y))
                    epoch_loss.assign_add(step_report_loss)
                    nstep_loss.assign_add(step_report_loss)
                    epoch_div.assign_add(step_toks)
                    nstep_div.assign_add(step_toks)

                    step = self.optimizer.global_step.numpy().item() + 1
                    if step % self.nsteps == 0:
                        metrics = self.calc_metrics(nstep_loss.numpy().item(), nstep_div.numpy().item())
                        self.report(
                            step, metrics, self.nstep_start,
                            'Train', 'STEP', reporting_fns, self.nsteps
                        )
                        nstep_loss.assign(0.0)
                        nstep_div.assign(0)
                        self.nstep_start = time.time()

                epoch_loss = epoch_loss.numpy()
                epoch_div = epoch_div.numpy()
                metrics = self.calc_metrics(epoch_loss, epoch_div)
                self.train_epochs += 1
                self.report(
                    self.train_epochs, metrics, start,
                    'Train', 'EPOCH', reporting_fns
                )
                return metrics

    def calc_metrics(self, agg, norm):
        metrics = super().calc_metrics(agg, norm)
        metrics['perplexity'] = np.exp(metrics['avg_loss']).item()
        return metrics

    def _evaluate(self, es, reporting_fns, **kwargs):
        """Run the model with beam search and report Bleu.

        :param es: `tf.dataset` of input
        :param reporting_fns: Input hooks
        """
        preds = []
        golds = []
        start = time.time()
        kwargs['make_input'] = False

        for features, tgt in es:
            tgt_lens = features.pop('tgt_len')
            top_preds = self.model.predict(features, **kwargs)
            preds.extend(convert_seq2seq_preds(top_preds[:, 0, :], self.tgt_rlut))
            golds.extend(convert_seq2seq_golds(tgt, tgt_lens, self.tgt_rlut))
        metrics = {'bleu': bleu(preds, golds, self.bleu_n_grams)[0]}
        self.report(
            0, metrics, start, 'Test', 'EPOCH', reporting_fns
        )
        return metrics

    def test(self, vs, reporting_fns, steps=0, phase='Valid', **kwargs):
        """Run an epoch of testing over the dataset

        If we are using a `tf.dataset`-based `fit_func`, we will just
        cycle the number of steps and let the `dataset` yield new batches.

        If we are using `feed_dict`s, we convert each batch from the `DataFeed`
        and pass that into TF as the `feed_dict`

        :param vs: A validation set
        :param reporting_fns: Reporting hooks
        :param phase: The phase of evaluation (`Test`, `Valid`)
        :param dataset: (`bool`) Are we using `tf.dataset`s
        :return: Metrics
        """

        def _replicated_valid_step(inputs):
            features, tgt = inputs
            top_preds = self.model.predict(features, beam=1, make_input=False)
            per_replica_loss = loss(self.model, features, tgt)
            per_replica_toks = self._num_toks(features['tgt_len'])
            per_replica_report_loss = per_replica_loss * tf.cast(per_replica_toks, tf.float32)
            return per_replica_report_loss, per_replica_toks, top_preds

        if phase == 'Test':
            SET_TRAIN_FLAG(False)
            return self._evaluate(vs, reporting_fns, **kwargs)

        strategy = self.strategy
        num_replicas = strategy.num_replicas_in_sync

        with strategy.scope():

            SET_TRAIN_FLAG(False)
            self.valid_epochs += 1

            total_loss = tf.Variable(0.0)
            total_toks = tf.Variable(0, dtype=tf.int32)
            preds = []
            golds = []

            start = time.time()

            test_iter = iter(vs)

            for i in range(steps):
                features, tgt = next(test_iter)
                inputs = (features, tgt)
                per_replica_loss, per_replica_toks, _ = strategy.run(_replicated_valid_step, args=(inputs,))
                total_loss.assign_add(strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None))
                total_toks.assign_add(strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_toks, axis=None))
                # Not sure a good way to get top preds merged yet

            metrics = self.calc_metrics(total_loss.numpy(), total_toks.numpy())
            self.report(
                self.valid_epochs, metrics, start,
                phase, 'EPOCH', reporting_fns
            )
            return metrics

    def distribute(self, dataset):
        return self.strategy.experimental_distribute_dataset(dataset)


@register_training_func('seq2seq', name="distributed")
def fit_eager_distributed(model_params, ts, vs, es=None, **kwargs):
    """
    Train an language model using TensorFlow with `tf.dataset`.  This
    is the default behavior for training.

    :param model_params: The model (or parameters to create the model) to train
    :param ts: A training data set
    :param vs: A validation data set
    :param es: A test data set, can be None
    :param kwargs:
        See below

    :Keyword Arguments:
        * *do_early_stopping* (``bool``) --
          Stop after evaluation data is no longer improving.  Defaults to True
        * *verbose* (`dict`) A dictionary containing `console` boolean and `file` name if on
        * *epochs* (``int``) -- how many epochs.  Default to 20
        * *outfile* -- Model output file, defaults to classifier-model.pyth
        * *patience* --
           How many epochs where evaluation is no longer improving before we give up
        * *reporting* --
           Callbacks which may be used on reporting updates
        * *nsteps* (`int`) -- If we should report every n-steps, this should be passed
        * *ema_decay* (`float`) -- If we are doing an exponential moving average, what decay to us4e
        * *clip* (`int`) -- If we are doing gradient clipping, what value to use
        * *optim* (`str`) -- The name of the optimizer we are using
        * *lr* (`float`) -- The learning rate we are using
        * *mom* (`float`) -- If we are using SGD, what value to use for momentum
        * *beta1* (`float`) -- Adam-specific hyper-param, defaults to `0.9`
        * *beta2* (`float`) -- Adam-specific hyper-param, defaults to `0.999`
        * *epsilon* (`float`) -- Adam-specific hyper-param, defaults to `1e-8

    :return: None
    """

    epochs = int(kwargs.get('epochs', 5))
    patience = int(kwargs.get('patience', epochs))

    model_file = get_model_file('seq2seq', 'tf', kwargs.get('basedir'))

    do_early_stopping = bool(kwargs.get('do_early_stopping', True))

    best_metric = 0
    if do_early_stopping:
        early_stopping_metric = kwargs.get('early_stopping_metric', 'perplexity')
        early_stopping_cmp, best_metric = get_metric_cmp(early_stopping_metric, kwargs.get('early_stopping_cmp'))
        patience = kwargs.get('patience', epochs)
        print('Doing early stopping on [%s] with patience [%d]' % (early_stopping_metric, patience))

    reporting_fns = listify(kwargs.get('reporting', []))
    print('reporting', reporting_fns)

    batchsz = kwargs['batchsz']
    test_batchsz = kwargs.get('test_batchsz', batchsz)
    tgt_key = model_params.get('tgt_key')

    src_lengths_key = model_params.get('src_lengths_key')
    train_dataset = tf.data.Dataset.from_tensor_slices(to_tensors(ts, src_lengths_key, dst=True))
    train_dataset = train_dataset.shuffle(buffer_size=SHUF_BUF_SZ)
    train_dataset = train_dataset.batch(batchsz, drop_remainder=True)
    train_dataset = train_dataset.prefetch(NUM_PREFETCH)

    valid_dataset = tf.data.Dataset.from_tensor_slices(to_tensors(vs, src_lengths_key, dst=True))
    valid_dataset = valid_dataset.batch(batchsz, drop_remainder=True)
    valid_dataset = valid_dataset.prefetch(NUM_PREFETCH)

    trainer = Seq2SeqTrainerDistributedTf(model_params, **kwargs)

    train_dataset = trainer.distribute(train_dataset)
    valid_dataset = trainer.distribute(valid_dataset)

    last_improved = 0
    SET_TRAIN_FLAG(True)

    for epoch in range(epochs):

        trainer.train(train_dataset, steps=len(ts.examples) // ts.batchsz, reporting_fns=reporting_fns)
        test_metrics = trainer.test(valid_dataset, steps=len(vs.examples) // vs.batchsz, reporting_fns=reporting_fns,
                                    phase='Valid')

        if do_early_stopping is False:
            trainer.checkpoint()
            trainer.model.save(model_file)

        elif early_stopping_cmp(test_metrics[early_stopping_metric], best_metric):
            last_improved = epoch
            best_metric = test_metrics[early_stopping_metric]
            print('New best %.3f' % best_metric)
            trainer.checkpoint()
            trainer.model.save(model_file)

        elif (epoch - last_improved) > patience:
            print('Stopping due to persistent failures to improve')
            break

    if do_early_stopping is True:
        print('Best performance on %s: %.3f at epoch %d' % (early_stopping_metric, best_metric, last_improved))

    if es is not None:
        print('Reloading best checkpoint')
        trainer.recover_last_checkpoint()
        test_dataset = tf.data.Dataset.from_tensor_slices(to_tensors(es, src_lengths_key, dst=True))
        test_dataset = test_dataset.batch(test_batchsz, drop_remainder=False)
        test_dataset = test_dataset.prefetch(NUM_PREFETCH)
        trainer.test(test_dataset, steps=len(es.examples) // es.batchsz, reporting_fns=reporting_fns, phase='Test')

