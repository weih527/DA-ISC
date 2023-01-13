'''
Description: 
Author: weihuang
Date: 2021-11-26 09:22:42
LastEditors: weihuang
LastEditTime: 2022-02-10 00:58:08
'''

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import yaml
import time
import logging
import argparse
import numpy as np
from attrdict import AttrDict
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from utils.utils import setup_seed
from dataset.target_dataset_mito import targetDataSet_val_twoimgs, Evaluation
from model.CoDetectionCNN import CoDetectionCNN
from model.discriminator_damtnet import labelDiscriminator, featureDiscriminator
from model.discriminator_davsn import get_fc_discriminator
from loss.loss import CrossEntropy2d, BCELoss
from utils.metrics import dice_coeff
from utils.show import show_training, save_prediction_image
from utils.utils import adjust_learning_rate, adjust_learning_rate_discriminator
from utils.utils import inference_results
from utils.utils import get_current_consistency_weight

import warnings
warnings.filterwarnings("ignore")

def init_project(cfg):
    print('Initialization ... ', end='', flush=True)
    t1 = time.time()
    def init_logging(path):
        logging.basicConfig(
                level    = logging.INFO,
                format   = '%(message)s',
                datefmt  = '%m-%d %H:%M',
                filename = path,
                filemode = 'w')

        # define a Handler which writes INFO messages or higher to the sys.stderr
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)

        # set a format which is simpler for console use
        formatter = logging.Formatter('%(message)s')
        # tell the handler to use this format
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    # seeds
    setup_seed(cfg.TRAIN.random_seed)
    if cfg.TRAIN.if_cuda:
        if torch.cuda.is_available() is False:
            raise AttributeError('No GPU available')
        torch.backends.cudnn.enabled = True
        # torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.deterministic = True

    prefix = cfg.time
    if cfg.TRAIN.resume:
        model_name = cfg.TRAIN.model_name
    else:
        model_name = prefix + '_' + cfg.NAME
    cfg.cache_path = os.path.join(cfg.TRAIN.cache_path, model_name)
    cfg.save_path = os.path.join(cfg.TRAIN.save_path, model_name)
    cfg.record_path = os.path.join(cfg.save_path, model_name)
    cfg.valid_path = os.path.join(cfg.save_path, 'valid')
    if cfg.TRAIN.resume is False:
        if not os.path.exists(cfg.cache_path):
            os.makedirs(cfg.cache_path)
        if not os.path.exists(cfg.save_path):
            os.makedirs(cfg.save_path)
        if not os.path.exists(cfg.record_path):
            os.makedirs(cfg.record_path)
        if not os.path.exists(cfg.valid_path):
            os.makedirs(cfg.valid_path)
    init_logging(os.path.join(cfg.record_path, prefix + '.log'))
    logging.info(cfg)
    writer = SummaryWriter(cfg.record_path)
    writer.add_text('cfg', str(cfg))
    print('Done (time: %.2fs)' % (time.time() - t1))
    return writer

def load_dataset(cfg):
    print('Caching datasets ... ', end='', flush=True)
    t1 = time.time()
    source_data = sourceDataSet(cfg.DATA.data_dir_img,
                                cfg.DATA.data_dir_label,
                                cfg.DATA.data_list,
                                crop_size=(cfg.DATA.input_size, cfg.DATA.input_size),
                                stride=cfg.DATA.source_stride)
    train_provider = torch.utils.data.DataLoader(source_data,
                                           batch_size=cfg.TRAIN.batch_size,
                                           shuffle=True,
                                           num_workers=cfg.TRAIN.num_workers)
    if cfg.TRAIN.if_valid:
        val_data = targetDataSet_val_twoimgs(cfg.DATA.data_dir_val,
                                            cfg.DATA.data_dir_val_label,
                                            cfg.DATA.data_list_val,
                                            crop_size=(cfg.DATA.input_size_test, cfg.DATA.input_size_test),
                                            stride=cfg.DATA.target_stride)
        valid_provider = torch.utils.data.DataLoader(val_data,
                                           batch_size=1,
                                           shuffle=False)
    else:
        valid_provider = None
    print('Done (time: %.2fs)' % (time.time() - t1))
    return train_provider, valid_provider

def build_model(cfg, writer):
    print('Building model on ', end='', flush=True)
    t1 = time.time()
    device = torch.device('cuda:0')
    model = CoDetectionCNN(n_channels=cfg.MODEL.input_nc,
                           n_classes=cfg.MODEL.output_nc).to(device)

    cuda_count = torch.cuda.device_count()
    if cuda_count > 1:
        if cfg.TRAIN.batch_size % cuda_count == 0:
            print('%d GPUs ... ' % cuda_count, end='', flush=True)
            model = nn.DataParallel(model)
        else:
            raise AttributeError('Batch size (%d) cannot be equally divided by GPU number (%d)' % (cfg.TRAIN.batch_size, cuda_count))
    else:
        print('a single GPU ... ', end='', flush=True)
    print('Done (time: %.2fs)' % (time.time() - t1))
    return model

def resume_params(cfg, model, optimizer, resume):
    if resume:
        t1 = time.time()
        model_path = os.path.join(cfg.save_path, 'model-%06d.ckpt' % cfg.TRAIN.model_id)

        print('Resuming weights from %s ... ' % model_path, end='', flush=True)
        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_weights'])
            # optimizer.load_state_dict(checkpoint['optimizer_weights'])
        else:
            raise AttributeError('No checkpoint found at %s' % model_path)
        print('Done (time: %.2fs)' % (time.time() - t1))
        print('valid %d' % checkpoint['current_iter'])
        return model, optimizer, checkpoint['current_iter']
    else:
        return model, optimizer, 0

# def calculate_lr(iters):
#     if iters < cfg.TRAIN.warmup_iters:
#         current_lr = (cfg.TRAIN.base_lr - cfg.TRAIN.end_lr) * pow(float(iters) / cfg.TRAIN.warmup_iters, cfg.TRAIN.power) + cfg.TRAIN.end_lr
#     else:
#         if iters < cfg.TRAIN.decay_iters:
#             current_lr = (cfg.TRAIN.base_lr - cfg.TRAIN.end_lr) * pow(1 - float(iters - cfg.TRAIN.warmup_iters) / cfg.TRAIN.decay_iters, cfg.TRAIN.power) + cfg.TRAIN.end_lr
#         else:
#             current_lr = cfg.TRAIN.end_lr
#     return current_lr

def loop(cfg, train_provider, valid_provider, model, optimizer, iters, writer):
    f_loss_txt = open(os.path.join(cfg.record_path, 'loss.txt'), 'a')
    f_loss_adv_txt = open(os.path.join(cfg.record_path, 'loss_adv.txt'), 'a')
    f_valid_txt = open(os.path.join(cfg.record_path, 'valid.txt'), 'a')
    rcd_time = []
    sum_time = 0.0
    sum_loss_supervised = 0.0
    sum_loss_cpred = 0.0
    sum_loss_apred = 0.0
    sum_loss_diff = 0.0
    sum_loss_adv = 0.0
    sum_loss_cross_target = 0.0
    sum_loss_adv_spatial = 0.0
    sum_loss_adv_temporal = 0.0
    target_stride = cfg.DATA.target_stride
    device = torch.device('cuda:0')
    
    # build dataloder for target dataset
    target_data = sourceDataSet(cfg.DATA.data_dir_target,
                                cfg.DATA.data_dir_target_label,
                                cfg.DATA.data_list_target,
                                crop_size=(cfg.DATA.input_size_target, cfg.DATA.input_size_target),
                                stride=cfg.DATA.target_stride)
    targetloader = torch.utils.data.DataLoader(target_data,
                                         batch_size=cfg.TRAIN.batch_size,
                                         shuffle=True,
                                         num_workers=cfg.TRAIN.num_workers)
    
    target_evaluation = Evaluation(root_label=cfg.DATA.data_dir_val_label,
                                   crop_size=(cfg.DATA.input_size_test, cfg.DATA.input_size_test))
    trainloader_iter = enumerate(train_provider)
    targetloader_iter = enumerate(targetloader)
    if_adv_weight = cfg.TRAIN.if_adv_weight
    
    model_spatial = get_fc_discriminator(num_classes=cfg.MODEL.num_classes).to(device)
    model_temporal = get_fc_discriminator(num_classes=cfg.MODEL.num_classes).to(device)
    model_spatial.train()
    model_temporal.train()
    
    # build optimizer for discriminator
    optimizer_model_spatial = optim.Adam(model_spatial.parameters(), lr=cfg.TRAIN.learning_rate_ms,
                                        betas=(0.9, 0.99))
    optimizer_model_temporal = optim.Adam(model_temporal.parameters(), lr=cfg.TRAIN.learning_rate_mt,
                                        betas=(0.9, 0.99))
    
    # build loss functions
    criterion_seg = CrossEntropy2d().to(device)
    criterion_adv = BCELoss()
    
    # labels for adversarial training
    source_label = 0
    target_label = 1
    
    while iters <= cfg.TRAIN.total_iters:
        iters += 1
        t1 = time.time()
        model.train()
        
        optimizer.zero_grad()
        optimizer_model_spatial.zero_grad()
        optimizer_model_temporal.zero_grad()
        
        # adapt LR if needed
        adjust_learning_rate(optimizer, iters, cfg.TRAIN.learning_rate, cfg.TRAIN.total_iters, cfg.TRAIN.power)
        adjust_learning_rate_discriminator(optimizer_model_spatial, iters, cfg.TRAIN.learning_rate_ms, cfg.TRAIN.total_iters, cfg.TRAIN.power)
        adjust_learning_rate_discriminator(optimizer_model_temporal, iters, cfg.TRAIN.learning_rate_mt, cfg.TRAIN.total_iters, cfg.TRAIN.power)
        
        # train G
        for param in model_spatial.parameters():
            param.requires_grad = False

        for param in model_temporal.parameters():
            param.requires_grad = False
        
        # train with source
        _, batch = trainloader_iter.__next__()
        cimg_source, clabel_source, aimg_source, alabel_source, dlabel_source = batch
        cimg_source = cimg_source.to(device)
        aimg_source = aimg_source.to(device)
        clabel_source = clabel_source.to(device)
        alabel_source = alabel_source.to(device)
        dlabel_source = dlabel_source.to(device)
        img_cat = torch.cat([cimg_source, aimg_source], dim=1)
        cpred_source, apred_source, dpred_source = model(img_cat)
        
        loss_cpred = criterion_seg(cpred_source, clabel_source.long())
        loss_apred = criterion_seg(apred_source, alabel_source.long())
        loss_diff = criterion_seg(dpred_source, dlabel_source.long())  #TODO focal loss 
        
        if cfg.TRAIN.consistency_weight_rampup:
            consistency_weight = get_current_consistency_weight(iters, consistency=cfg.TRAIN.weight_cross, consistency_rampup=cfg.TRAIN.rampup_iters)
        else:
            consistency_weight = cfg.TRAIN.weight_cross
        
        if cfg.TRAIN.cross_loss_source:
            cpred_source_detach = cpred_source.clone().detach()
            cpred_source_detach = torch.argmax(cpred_source_detach, dim=1)
            apred_source_detach = apred_source.clone().detach()
            apred_source_detach = torch.argmax(apred_source_detach, dim=1)
            dpred_source_detach = dpred_source.clone().detach()
            dpred_source_detach = torch.argmax(dpred_source_detach, dim=1)
            clabel_source_cross = torch.abs(apred_source_detach - dpred_source_detach)
            alabel_source_cross = torch.abs(cpred_source_detach - dpred_source_detach)
            loss_cpred_cross = criterion_seg(cpred_source, clabel_source_cross.long()) * consistency_weight
            loss_apred_cross = criterion_seg(apred_source, alabel_source_cross.long()) * consistency_weight
            loss = loss_cpred + loss_apred + loss_diff + loss_cpred_cross + loss_apred_cross
            sum_loss_cpred += loss_cpred_cross.item()
            sum_loss_apred += loss_apred_cross.item()
        else:
            loss = loss_cpred + loss_apred + loss_diff
            sum_loss_cpred += loss_cpred.item()
            sum_loss_apred += loss_apred.item()
        loss.backward()
        sum_loss_supervised += loss.item()
        sum_loss_diff += loss_diff.item()
        
        # train with target
        _, batch = targetloader_iter.__next__()
        cimg_target, _, aimg_target, _, _ = batch
        cimg_target = cimg_target.to(device)
        aimg_target = aimg_target.to(device)
        img_cat_target = torch.cat([cimg_target, aimg_target], dim=1)
        cpred_target, apred_target, dpred_target = model(img_cat_target)
        if cfg.TRAIN.cross_loss_target:
            cpred_target_detach = cpred_target.clone().detach()
            cpred_target_detach = torch.argmax(cpred_target_detach, dim=1)
            apred_target_detach = apred_target.clone().detach()
            apred_target_detach = torch.argmax(apred_target_detach, dim=1)
            dpred_target_detach = dpred_target.clone().detach()
            dpred_target_detach = torch.argmax(dpred_target_detach, dim=1)
            clabel_target_cross = torch.abs(apred_target_detach - dpred_target_detach)
            alabel_target_cross = torch.abs(cpred_target_detach - dpred_target_detach)
            loss_cpred_cross_target = criterion_seg(cpred_target, clabel_target_cross.long()) * consistency_weight
            loss_apred_cross_target = criterion_seg(apred_target, alabel_target_cross.long()) * consistency_weight
            loss = loss_cpred_cross_target + loss_apred_cross_target
            sum_loss_cross_target += loss.item()
        else:
            loss = 0.0
            sum_loss_cross_target = 0.0
        
        # Target Domain Adv loss
        # spatia
        cadv_out_spatial = model_spatial(F.softmax(cpred_target))
        loss_cadv_spatial = criterion_adv(cadv_out_spatial, source_label)
        aadv_out_spatial = model_spatial(F.softmax(apred_target))
        loss_aadv_spatial = criterion_adv(aadv_out_spatial, source_label)
        loss_adv_spatial = cfg.TRAIN.weight_adv_spatial * (loss_cadv_spatial + loss_aadv_spatial) / 2.0
        # temporal adv
        adv_out_temporal = model_temporal(F.softmax(dpred_target))
        loss_adv_temporal = criterion_adv(adv_out_temporal, source_label)
        loss_adv_temporal = cfg.TRAIN.weight_adv_temporal * loss_adv_temporal
        loss = loss + loss_adv_temporal + loss_adv_spatial
        loss.backward()
        sum_loss_adv += loss.item()
        sum_loss_adv_spatial += loss_adv_spatial.item()
        sum_loss_adv_temporal += loss_adv_temporal.item()
        
        # train discriminator
        for param in model_temporal.parameters():
            param.requires_grad = True
        for param in model_spatial.parameters():
            param.requires_grad = True
        
        # Train with source
        # spatial
        cpred_source_spatial = cpred_source.detach()
        adv_out_spatial = model_spatial(F.softmax(cpred_source_spatial))
        loss_adv_spatial = criterion_adv(adv_out_spatial, source_label) / 4
        loss_adv_spatial.backward()
        apred_source_spatial = apred_source.detach()
        adv_out_spatial = model_spatial(F.softmax(apred_source_spatial))
        loss_adv_spatial = criterion_adv(adv_out_spatial, source_label) / 4
        loss_adv_spatial.backward()
        # temporal
        dpred_source_temporal = dpred_source.detach()
        adv_out_temporal = model_temporal(F.softmax(dpred_source_temporal))
        loss_adv_temporal = criterion_adv(adv_out_temporal, source_label) / 2
        loss_adv_temporal.backward()
        
        # Train with target
        # spatial
        cpred_target_spatial = cpred_target.detach()
        adv_out_spatial = model_spatial(F.softmax(cpred_target_spatial))
        loss_adv_spatial = criterion_adv(adv_out_spatial, target_label) / 4
        loss_adv_spatial.backward()
        apred_target_spatial = apred_target.detach()
        adv_out_spatial = model_spatial(F.softmax(apred_target_spatial))
        loss_adv_spatial = criterion_adv(adv_out_spatial, target_label) / 4
        loss_adv_spatial.backward()
        # temporal
        dpred_target_temporal = dpred_target.detach()
        adv_out_temporal = model_temporal(F.softmax(dpred_target_temporal))
        loss_adv_temporal = criterion_adv(adv_out_temporal, target_label) / 2
        loss_adv_temporal.backward()
        
        if if_adv_weight:
            # Discriminators' weights discrepancy (wd)
            k = 0
            loss_wd = 0
            for (W1, W2) in zip(model_temporal.parameters(), model_spatial.parameters()):
                W1 = W1.view(-1)
                W2 = W2.view(-1)
                loss_wd = loss_wd + (torch.matmul(W1, W2) / (torch.norm(W1) * torch.norm(W2)) + 1)
                k += 1
            loss_wd = loss_wd / k
            loss = cfg.TRAIN.lamda_wd * loss_wd
            loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        torch.nn.utils.clip_grad_norm_(model_temporal.parameters(), 1)
        torch.nn.utils.clip_grad_norm_(model_spatial.parameters(), 1)
        
        optimizer.step()
        optimizer_model_spatial.step()
        optimizer_model_temporal.step()
        learning_rate = optimizer.param_groups[0]['lr']
        
        sum_time += time.time() - t1
        
        # log train
        if iters % cfg.TRAIN.display_freq == 0 or iters == 1:
            rcd_time.append(sum_time)
            if iters == 1:
                logging.info('step %d, loss=%.6f, loss_cpred=%.6f, loss_apred=%.6f, loss_diff=%.6f (wt: *1, lr: %.8f, et: %.2f sec, rd: %.2f min)'
                            % (iters, sum_loss_supervised, sum_loss_cpred, sum_loss_apred, sum_loss_diff, learning_rate, sum_time,
                            (cfg.TRAIN.total_iters - iters) / cfg.TRAIN.display_freq * np.mean(np.asarray(rcd_time)) / 60))
                logging.info('step %d, loss=%.6f, loss_cross=%.6f, loss_adv_spatial=%.6f, loss_adv_temporal=%.6f'
                            % (iters, sum_loss_adv, sum_loss_cross_target, sum_loss_adv_spatial, sum_loss_adv_temporal))
            else:
                logging.info('step %d, loss=%.6f, loss_cpred=%.6f, loss_apred=%.6f, loss_diff=%.6f (wt: *1, lr: %.8f, et: %.2f sec, rd: %.2f min)'
                            % (iters, \
                                sum_loss_supervised / cfg.TRAIN.display_freq, \
                                sum_loss_cpred / cfg.TRAIN.display_freq, \
                                sum_loss_apred / cfg.TRAIN.display_freq, \
                                sum_loss_diff / cfg.TRAIN.display_freq, learning_rate, sum_time, \
                            (cfg.TRAIN.total_iters - iters) / cfg.TRAIN.display_freq * np.mean(np.asarray(rcd_time)) / 60))
                logging.info('step %d, loss=%.6f, loss_cross=%.6f, loss_adv_spatial=%.6f, loss_adv_temporal=%.6f'
                            % (iters, \
                                sum_loss_adv / cfg.TRAIN.display_freq, \
                                sum_loss_cross_target / cfg.TRAIN.display_freq, \
                                sum_loss_adv_spatial / cfg.TRAIN.display_freq, \
                                sum_loss_adv_temporal / cfg.TRAIN.display_freq))
                writer.add_scalar('loss_supervised', sum_loss_supervised / cfg.TRAIN.display_freq, iters)
                writer.add_scalar('loss_adv', sum_loss_adv / cfg.TRAIN.display_freq, iters)
                f_loss_txt.write('step=%d, loss=%.6f, loss_cpred=%.6f, loss_apred=%.6f, loss_diff=%.6f' % \
                    (iters, sum_loss_supervised / cfg.TRAIN.display_freq, \
                                sum_loss_cpred / cfg.TRAIN.display_freq, \
                                sum_loss_apred / cfg.TRAIN.display_freq, \
                                sum_loss_diff / cfg.TRAIN.display_freq))
                f_loss_txt.write('\n')
                f_loss_txt.flush()
                f_loss_adv_txt.write('step=%d, loss=%.6f, loss_cross=%.6f, loss_adv_spatial=%.6f, loss_adv_temporal=%.6f' % \
                    (iters, sum_loss_adv / cfg.TRAIN.display_freq, 
                            sum_loss_cross_target / cfg.TRAIN.display_freq, 
                            sum_loss_adv_spatial / cfg.TRAIN.display_freq, 
                            sum_loss_adv_temporal / cfg.TRAIN.display_freq))
                f_loss_adv_txt.write('\n')
                f_loss_adv_txt.flush()
                sys.stdout.flush()
                sum_time = 0.0
                sum_loss_supervised = 0.0
                sum_loss_cpred = 0.0
                sum_loss_apred = 0.0
                sum_loss_diff = 0.0
                sum_loss_adv = 0.0
                sum_loss_cross_target = 0.0
                sum_loss_adv_spatial = 0.0
                sum_loss_adv_temporal = 0.0
        
        # display
        if iters % cfg.TRAIN.show_freq == 0 or iters == 1:
            show_training(iters, cimg_source[0], clabel_source[0], cpred_source[0], cfg.cache_path, tag='c')
            # show_training(iters, aimg_source[0], alabel_source[0], apred_source[0], cfg.cache_path, tag='a')
            # show_training(iters, cimg_source[0], dlabel_source[0], dpred_source[0], cfg.cache_path, tag='d')
        
        # valid
        if cfg.TRAIN.if_valid:
            if iters % cfg.TRAIN.valid_freq == 0 or iters == 1:
                model.eval()
                preds = np.zeros((100, cfg.DATA.input_size_test, cfg.DATA.input_size_test), dtype=np.uint8)
                for i_pic, (cimg, aimg) in enumerate(valid_provider):
                    cimg = cimg.to(device)
                    aimg = aimg.to(device)
                    img_cat = torch.cat([cimg, aimg], dim=1)
                    with torch.no_grad():
                        cpred, apred, _ = model(img_cat)
                    preds[i_pic] = inference_results(cpred, preds[i_pic], mode='mito')
                    preds[i_pic+target_stride] = inference_results(apred, preds[i_pic+target_stride], mode='mito')
                F1 = target_evaluation(preds)
                logging.info('model-%d, F1=%.6f' % (iters, F1))
                writer.add_scalar('valid/F1', F1, iters)
                f_valid_txt.write('model-%d, F1=%.6f' % (iters, F1))
                f_valid_txt.write('\n')
                f_valid_txt.flush()
                torch.cuda.empty_cache()
        
        # save
        if iters % cfg.TRAIN.save_freq == 0:
            states = {'current_iter': iters, 'valid_result': None,
                    'model_weights': model.state_dict()}
            torch.save(states, os.path.join(cfg.save_path, 'model-%06d.ckpt' % iters))
            print('***************save modol, iters = %d.***************' % (iters), flush=True)
    f_loss_txt.close()
    f_loss_adv_txt.close()
    f_valid_txt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='standard', help='path to config file')
    parser.add_argument('-m', '--mode', type=str, default='train', help='path to config file')
    args = parser.parse_args()

    cfg_file = args.cfg + '.yaml'
    print('cfg_file: ' + cfg_file)
    print('mode: ' + args.mode)

    with open('./config/' + cfg_file, 'r') as f:
        cfg = AttrDict(yaml.load(f, Loader=yaml.FullLoader))

    timeArray = time.localtime()
    time_stamp = time.strftime('%Y-%m-%d--%H-%M-%S', timeArray)
    print('time stamp:', time_stamp)
    cfg.path = cfg_file
    cfg.time = time_stamp

    try:
        if cfg.DATA.aug_chang:
            from dataset.source_dataset_mito import sourceDataSet_chang as sourceDataSet
            print('Import sourceDataSet_chang')
        else:
            from dataset.source_dataset_mito import sourceDataSet
    except:
        from dataset.source_dataset_mito import sourceDataSet
    
    if args.mode == 'train':
        writer = init_project(cfg)
        train_provider, valid_provider = load_dataset(cfg)
        model = build_model(cfg, writer)
        if cfg.TRAIN.opt_type == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.learning_rate, betas=(0.9, 0.99))
        else:
            optimizer = optim.SGD(model.parameters(),
                                lr=cfg.TRAIN.learning_rate,
                                momentum=0.9,
                                weight_decay=0.0005)
        model, optimizer, init_iters = resume_params(cfg, model, optimizer, cfg.TRAIN.resume)
        loop(cfg, train_provider, valid_provider, model, optimizer, init_iters, writer)
        writer.close()
    else:
        raise NotImplementedError
    print('***Done***')