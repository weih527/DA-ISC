'''
Description: 
Author: weihuang
Date: 2021-11-18 15:47:44
LastEditors: Please set LastEditors
LastEditTime: 2021-11-29 17:54:38
'''
import torch
import random
import numpy as np

def setup_seed(seed):
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def lr_poly(base_lr, iter, max_iter, power):
    """ Poly_LR scheduler
    """
    return base_lr * ((1 - float(iter) / max_iter) ** power)

def _adjust_learning_rate(optimizer, i_iter, learning_rate, max_iters, power):
    lr = lr_poly(learning_rate, i_iter, max_iters, power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def adjust_learning_rate(optimizer, i_iter, learning_rate, max_iters, power):
    """ adject learning rate for main segnet
    """
    _adjust_learning_rate(optimizer, i_iter, learning_rate, max_iters, power)

def adjust_learning_rate_discriminator(optimizer, i_iter, learning_rate, max_iters, power):
    _adjust_learning_rate(optimizer, i_iter, learning_rate, max_iters, power)

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def get_current_consistency_weight(epoch, consistency=0.1, consistency_rampup=10000.0):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return consistency * sigmoid_rampup(epoch, consistency_rampup)

def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
    n, c, h, w = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)

def inference_results(pred, previous, mode='lucchi'):
    pred = torch.argmax(pred, dim=1).float()
    pred = pred.data.cpu().numpy()
    pred = np.squeeze(pred)
    if mode == 'lucchi':
        pred = pred[176:-176, 48:-48]
    
    temp_cpred = previous.copy()
    if np.sum(previous) == 0:
        temp_cpred += pred.astype(np.uint8)
    else:
        temp_cpred += pred.astype(np.uint8)
        temp_cpred = temp_cpred.astype(np.float32) / 2.0
        temp_cpred[temp_cpred >= 0.5] = 1
        temp_cpred[temp_cpred < 0.5] = 0
    temp_cpred = temp_cpred.astype(np.uint8)
    return temp_cpred

def inference_results2(pred, previous, mode='lucchi'):
    # pred = torch.argmax(pred, dim=1).float()
    pred = torch.nn.functional.softmax(pred, dim=1)
    pred = pred[:, 1]
    pred = pred.data.cpu().numpy()
    pred = np.squeeze(pred)
    if mode == 'lucchi':
        pred = pred[176:-176, 48:-48]
    
    temp_cpred = previous.copy()
    if np.sum(previous) == 0:
        temp_cpred += pred
    else:
        temp_cpred += pred
        temp_cpred = temp_cpred / 2.0
    temp_cpred[temp_cpred<0] = 0
    temp_cpred[temp_cpred>1] = 1
    return temp_cpred