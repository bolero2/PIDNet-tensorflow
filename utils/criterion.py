# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------
import tensorflow as tf
import numpy as np
from tensorflow.keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.layers import Softmax
from tensorflow.experimental.numpy import ascontiguousarray
from configs import config


class CrossEntropy(tf.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.weight = weight
        self.criterion = SparseCategoricalCrossentropy()

    def _forward(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        loss = self.criterion(y_true, y_pred).numpy()

        return loss

    def __call__(self, y_true, y_pred):
        if config.MODEL.NUM_OUTPUTS == 1:
            y_pred = [y_pred]

        balance_weights = config.LOSS.BALANCE_WEIGHTS
        sb_weights = config.LOSS.SB_WEIGHTS
        if len(balance_weights) == len(y_pred):
            return sum([w * self._forward(y_true, x) for (w, x) in zip(balance_weights, y_pred)])
        elif len(y_pred) == 1:
            return sb_weights * self._forward(y_true, y_pred[0])
        
        else:
            raise ValueError("lengths of prediction and y_true are not identical!")

        


class OhemCrossEntropy(tf.Module):
    def __init__(self, ignore_label=-1, thres=0.7, min_kept=100000, weight=None):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label

        self.criterion = CategoricalCrossentropy(
            sample_weight=weight,
            reduction=tf.keras.losses.Reduction.NONE
        )

    def _ce_forward(self, score, target):

        loss = self.criterion(score, target)

        return loss

    def _ohem_forward(self, score, target, **kwargs):
        # print("Debugging start!")

        pred = Softmax(score, axis=1)
        # print("\n\nPRED 1:", pred.shape)
        
        pixel_losses = tf.reshape(ascontiguousarray(self.criterion(score, target)), -1)
        mask = tf.reshape(ascontiguousarray(target), -1) != self.ignore_label
        # print("Mask:", mask)
        # print("Mask:", np.unique(mask.detach().cpu().numpy()))

        tmp_target = target.clone()
        # print(tmp_target.shape)
        tmp_target[tmp_target == self.ignore_label] = 0
        # print(tmp_target.unsqueeze(1).shape)
        pred = pred.gather(1, tf.expand_dims(tmp_target, 3))
        # print('PRED 2:', pred.shape)
        pred, ind = ascontiguousarray(tf.reshape(ascontiguousarray(pred), (-1, ))[mask]).sort()
        # print('PRED 3:', pred, pred.shape)
        # print("HERE :", self.min_kept, pred.shape, pred.numel() - 1)

        min_value = pred[min(self.min_kept, pred.numel() - 1)]
            
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()

    def __call__(self, score, target):
        
        if not (isinstance(score, list) or isinstance(score, tuple)):
            score = [score]

        balance_weights = config.LOSS.BALANCE_WEIGHTS
        sb_weights = config.LOSS.SB_WEIGHTS
        if len(balance_weights) == len(score):
            functions = [self._ce_forward] * \
                (len(balance_weights) - 1) + [self._ohem_forward]
            return sum([
                w * func(x, target)
                for (w, x, func) in zip(balance_weights, score, functions)
            ])
        
        elif len(score) == 1:
            return sb_weights * self._ohem_forward(score[0], target)
        
        else:
            raise ValueError("lengths of prediction and target are not identical!")


def weighted_bce(bd_pre, target):
    n, c, h, w = tuple(bd_pre.shape)
    # log_p = bd_pre.permute(0, 2, 3, 1).contiguous().view(1, -1)
    log_p = tf.reshape(ascontiguousarray(bd_pre), (1, -1))
    target_t = tf.reshape(target, (1, -1))

    pos_index = (target_t == 1)
    neg_index = (target_t == 0)

    weight = tf.zeros_like(log_p)
    pos_num = np.sum(pos_index)#.sum()
    neg_num = np.sum(neg_index)#.sum()
    print("pos_num :", pos_num)
    print("neg_num :", neg_num)
    sum_num = pos_num + neg_num
    print(sum_num)
    weight[pos_index] = neg_num * 1.0 / sum_num
    weight[neg_index] = pos_num * 1.0 / sum_num

    loss = BinaryCrossentropy(log_p, target_t, weight, reduction=tf.keras.losses.Reduction.AUTO)

    return loss


class BondaryLoss(tf.Module):
    def __init__(self, coeff_bce = 20.0):
        super(BondaryLoss, self).__init__()
        self.coeff_bce = coeff_bce
        
    def __call__(self, bd_pre, bd_gt):
        print(bd_pre, bd_gt)

        bce_loss = self.coeff_bce * weighted_bce(bd_pre, bd_gt)
        loss = bce_loss
        
        return loss
    
if __name__ == '__main__':
    a = tf.zeros(2,64,64)
    a[:,5,:] = 1

    pre = tf.random.normal((2,1,16,16))
    
    Loss_fc = BondaryLoss()
    loss = Loss_fc(pre, a.to(tf.uint8))

        
        
        


