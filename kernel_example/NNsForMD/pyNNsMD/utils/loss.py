"""
Functions and classes for loss.

Also includes Metrics and tools around loss.
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
from tensorflow.keras import backend as K


def mask_MeanSquaredError(y_true, y_pred):
    mask = K.cast(K.not_equal(y_true, 0.0), dtype="float32")
    return K.sum(K.square(y_true*mask - y_pred*mask))/(K.sum(mask))


class ScaledMeanAbsoluteError(tf.keras.metrics.MeanAbsoluteError):

    def __init__(self, scaling_shape=(), name='mean_absolute_error', **kwargs):
        super(ScaledMeanAbsoluteError, self).__init__(name=name, **kwargs)
        self.scale = self.add_weight(shape=scaling_shape, initializer=tf.keras.initializers.Ones(), name='scale_mae',
                                     dtype=tf.keras.backend.floatx())
        self.scaling_shape = scaling_shape

    def reset_states(self):
            # Super variables
            #ks.backend.set_value(self.total, 0)
            #ks.backend.set_value(self.count, 0)
            self.total.assign(0)
            self.count.assign(0)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = self.scale * y_true
        y_pred = self.scale * y_pred
        return super(ScaledMeanAbsoluteError, self).update_state(y_true, y_pred, sample_weight=sample_weight)

    def get_config(self):
        """Returns the serializable config of the metric."""
        mae_conf = super(ScaledMeanAbsoluteError, self).get_config()
        mae_conf.update({"scaling_shape": self.scaling_shape})
        return mae_conf

    def set_scale(self,scale):
        #ks.backend.set_value(self.scale, scale)
        self.scale.assign(scale)


class MaskedScaledMeanAbsoluteError(tf.keras.metrics.Metric):
    def __init__(self, scaling_shape=(), name='mean_absolute_error', **kwargs):
        super(MaskedScaledMeanAbsoluteError, self).__init__(name=name, **kwargs)
        self.scale = self.add_weight(shape=scaling_shape, initializer=tf.keras.initializers.Ones(), name='scale_mae',
                                     dtype=tf.keras.backend.floatx())
        self.scaling_shape = scaling_shape
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def reset_states(self):
            # Super variables
            #ks.backend.set_value(self.total, 0)
            #ks.backend.set_value(self.count, 0)
            self.total.assign(0)
            self.count.assign(0)

    def update_state(self, y_true, y_pred, sample_weight=None):
        mask = K.cast(K.not_equal(y_true, 0.0), dtype="float32")
        y_true = self.scale * y_true
        y_pred = self.scale * y_pred
        self.total.assign_add(K.sum(K.abs(y_true - y_pred) * mask))
        self.count.assign_add(K.sum(mask))
        #return super(ScaledMeanAbsoluteError, self).update_state(y_true, y_pred, sample_weight=sample_weight)

    def result(self):
        return self.total/self.count

    def get_config(self):
        """Returns the serializable config of the metric."""
        mae_conf = super(MaskedScaledMeanAbsoluteError, self).get_config()
        mae_conf.update({"scaling_shape": self.scaling_shape})
        return mae_conf

    def set_scale(self,scale):
        #ks.backend.set_value(self.scale, scale)
        self.scale.assign(scale)


class ZeroEmptyLoss(tf.keras.losses.Loss):
    """
    Empty constant zero loss.
    """
    def __init__(self,**kwargs):
        self.zero_empty = tf.constant(0)
        super(ZeroEmptyLoss,self).__init__(**kwargs)

    def call(self, y_true, y_pred):
        """
        Returns:
            tf.constant(0)
        """
        return self.zero_empty


def get_lr_metric(optimizer):
    """
    Obtian learning rate from optimizer.

    Args:
        optimizer (tf.kears.optimizer): Optimizer used for training.

    Returns:
        float: learning rate.

    """

    def lr(y_true, y_pred):
        return optimizer.learning_rate

    return lr


def r2_metric(y_true, y_pred):
    """
    Compute r2 metric.

    Args:
        y_true (tf.tensor): True y-values.
        y_pred (tf.tensor): Predicted y-values.

    Returns:
        tf.tensor: r2 metric.

    """
    ss_res = ks.backend.sum(ks.backend.square(y_true - y_pred))
    ss_tot = ks.backend.sum(ks.backend.square(y_true - ks.backend.mean(y_true)))
    return 1 - ss_res / (ss_tot + ks.backend.epsilon())


def masked_r2_metric(y_true, y_pred):
    """
    Compute r2 metric with missing ground truth values.

    Args:
        y_true (tf.tensor): True y-values.
        y_pred (tf.tensor): Predicted y-values.

    Returns:
        tf.tensor: r2 metric.

    """
    mask = K.cast(K.not_equal(y_true, 0.0), dtype="float32")
    ss_res = ks.backend.sum(ks.backend.square(y_true * mask - y_pred * mask))
    ss_tot = K.sum(K.square((y_true * mask - K.sum(y_true * mask)/K.sum(mask)) * mask))
    return 1 - ss_res / (ss_tot + ks.backend.epsilon())


def merge_hist(hist1, hist2):
    """
    Merge two hist-dicts.

    Args:
        hist1 (dict): Hist dict from fit.
        hist2 (dict): Hist dict from fit.

    Returns:
        outhist (dict): hist1 + hist2.

    """
    outhist = {}
    for x, y in hist1.items():
        outhist.update({x: hist1[x] + hist2[x]})
    return outhist



class NACphaselessLoss(ks.losses.Loss):
    def __init__(self, name='phaseless_loss', number_state=2, shape_nac=(1, 1), **kwargs):
        super().__init__(name=name, **kwargs)
        self.number_state = number_state
        self.shape_nac = shape_nac

        phase_stat = np.array([[1], [-1]])
        for i in range(number_state - 1):
            temp_len = len(phase_stat)
            temp = np.expand_dims(np.array([1] * temp_len + [-1] * temp_len), axis=-1)
            phase_stat = np.concatenate([phase_stat, phase_stat], axis=0)
            phase_stat = np.concatenate([phase_stat, temp], axis=-1)
        # print("Possible phase combinations",phase_stat)

        phase_stat_mat = np.expand_dims(phase_stat, axis=1) * np.expand_dims(phase_stat, axis=-1)
        idxs = np.triu_indices(number_state, k=1)
        phase_end = np.unique(phase_stat_mat[..., idxs[0], idxs[1]], axis=0)
        # print("Final phase combinations",phase_end)
        # print("Length:", len(phase_end),2**(number_state-1))

        # Expand for broadcasting: batch + shape of nac
        phase_end = np.expand_dims(phase_end, axis=1)  # batch
        for i in range(len(shape_nac)):
            phase_end = np.expand_dims(phase_end, axis=-1)  # shape nac
        # print("Shape phase combinations",phase_end.shape)

        # Store phase factors
        self.phase_combo = tf.constant(phase_end, dtype=tf.keras.backend.floatx())
        self.reduce_mean = list(range(1, len(shape_nac) + 3))

    def call(self, y_true, y_pred):
        # Each phase combination should be along axis 0, batch at axis=1, state-state at axis=2 for mse
        # The number of state-state is about state*(state-1)/2
        # The number of combinations is 2^(state-1)
        phase_combo = tf.cast(self.phase_combo, y_pred.dtype)
        se = ks.backend.square(
            ks.backend.expand_dims(y_true, axis=0) - phase_combo * ks.backend.expand_dims(y_pred, axis=0))
        mse = ks.backend.mean(se, axis=self.reduce_mean)
        # Softmin
        # softmin = ks.backend.exp(-mse)
        # softmin = softmin/ks.backend.sum(softmin)
        # out = ks.backend.sum(softmin*mse)
        out = ks.backend.min(mse, axis=0)
        return out

    def get_config(self):
        """Return the config dictionary for a `Loss` instance."""
        return {'number_state': self.number_state,
                'shape_nac': self.shape_nac,
                'name': self.name}
