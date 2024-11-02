# -*- coding: cp949 -*-
import time
import os
import numpy as np
import argparse

import builtins
import warnings
import Levenshtein as Lev

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F

summary = None
import random
from data_loader import *

from transformers import AdamW
#train.cumulative_batch_count = 0

from functools import lru_cache
############
# CONSTANT #
############

from glob import glob

from model import CPCModel
    

def path_loader(root_path):
    
    file_list = sorted(glob(os.path.join(root_path, '*.npy')))
    
    return file_list


def get_asr_args():
    parser = argparse.ArgumentParser(description='Downstream hyperparameters')
    #hyperparams
    parser.add_argument('--device', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=2891)
    parser.add_argument('--ckpt', type=str, default='0')
    
    # ASR model hyperparameters 
    parser.add_argument('--mode', type=str, default=None, choices=['2dcnn', '1dcnn', 'linear', None, '2dcnn_ver2'])
    parser.add_argument('--input_size', type=int, default=768)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--anneal', default=1.1, type=float, help='Annealing learning rate every epoch')
    # etc
        
    # ML system parameters
    parser.add_argument('--workers', default=0, type=int, metavar='N', help='number of data loading workers (default: 32)')
    parser.add_argument('--gpu', type=str, default='0')        
    parser.add_argument('--resume', default='None', type=str, metavar='PATH', help='path to downstream checkpoint (default: none)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--print-freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--data_path', default='None', type=str, metavar='PATH', help='path to downstream checkpoint (default: none)')
    parser.add_argument('--percentage', default=100, type=int, help='percentage for randomly selecting training sets')
    parser.add_argument('--tensorboard_log', default='None', type=str, metavar='PATH', help='path to downstream checkpoint (default: none)')
       
    
    args = parser.parse_args()
    
    new_args = dict(device=args.device, batch_size=args.batch_size, epochs=args.epochs, lr=args.lr, seed=args.seed, ckpt=args.ckpt, mode=args.mode, input_size=args.input_size, d_model=args.d_model,  
        anneal=args.anneal, workers=args.workers, gpu=args.gpu, downstream_resume=args.resume, start_epoch=args.start_epoch, print_freq=args.print_freq, data_path=args.data_path, percentage=args.percentage, tensorboard_log=args.tensorboard_log)
    
    return new_args

def main():

    args = get_asr_args()
    
    # Fix seed and make backends deterministic    
    if args['seed'] is not None:
        random.seed(args['seed'])
        np.random.seed(args['seed'])
        torch.manual_seed(args['seed'])
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training.'
            'This will turn on the CUDNN deterministic setting,'
            'which can slow down your training considerably!'
            'You may see unexpected behavior when restarting'
            'from checkpoints.')
    
    
    device = torch.device("cuda:{}".format(args['gpu']))
    args['device'] = device
    print("Use GPU: {} for training".format(args['device']))
        
    torch.cuda.set_device(args['device'])
    
    cudnn.benchmark = True
    
    train_file_list = path_loader(os.path.join(args['data_path'], 'train'))
    print('all train_file_list {}'.format(len(train_file_list)))
    
    if args['percentage'] != 100:
        sample_size = max(1, int(len(train_file_list) * args['percentage'] * 0.01))
        indices = random.sample(range(len(train_file_list)), sample_size)
        
        train_file_list = [train_file_list[i] for i in indices]
        print('Percentage is not 100%. All train_file_list {} '.format(len(train_file_list)))
    
    val_file_list = path_loader(os.path.join(args['data_path'], 'test'))
    print('all val_file_list {}'.format(len(val_file_list)))
    
    model = CPCModel(input_channels=args['input_size'], encoder_output_dim=args['d_model'], mode=args['mode']).cuda()
    print('model is', model)    
    
    optimizer = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=1e-5)
    
    # optionally resume from a downstream checkpoint
    if args['downstream_resume']:
        if os.path.isfile(args['downstream_resume']):
            print("=> loading checkpoint '{}'".format(args['downstream_resume']))
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(args['device'])
            checkpoint = torch.load(args['downstream_resume'], map_location=loc)
            args['start_epoch'] = checkpoint['epoch']
            #pretrain_model.module.load_state_dict(checkpoint['TransformerModel'])
            model.load_state_dict(checkpoint['Model'])
            
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args['downstream_resume'], checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args['downstream_resume']))
    
    train_dataset = StaticFeatureDataset(path_list=train_file_list)
    valid_dataset = StaticFeatureDataset(path_list=val_file_list)
    
    #train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)        
    #valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)
    train_sampler = BucketingSampler(train_dataset, batch_size=args['batch_size'])
    
    '''
    train_loader = AudioDataLoader(train_dataset, batch_size=args['batch_size'], num_workers=sys_config['workers'], pin_memory=True, sampler=train_sampler)
    train_loader = AudioDataLoader(train_dataset, num_workers=args.workers, batch_sampler=train_sampler, pin_memory=True)
    
    valid_loader = AudioDataLoader(valid_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=sys_config['workers'], pin_memory=True, sampler=valid_sampler)
    valid_loader = AudioDataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    '''

    train_loader = AudioDataLoader(train_dataset, num_workers=args['workers'], batch_sampler=train_sampler, pin_memory=True)    
    valid_loader = AudioDataLoader(valid_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=args['workers'], pin_memory=True)
    
    print('Training mode! Data loading finished')
        
    train_batch_num = len(train_loader)
    print('Train Batch Num ', train_batch_num)
    valid_batch_num = len(valid_loader)
    print('Valid Batch Num ', valid_batch_num)
    
    save_dir = './models/ckpt={}'.format(args['ckpt'])
    
    from tensorboardX import SummaryWriter
    global summary
    if args['tensorboard_log'] is not None:
        summary = SummaryWriter(args['tensorboard_log'])
        print('summary loaded from {}'.format(args['tensorboard_log']))
    else:
        summary = SummaryWriter()
        
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)       
    
        
    train_begin = time.time()
    for epoch in range(args['start_epoch'], args['epochs']):
        save_epoch = epoch+1
        save_model_name = os.path.join(save_dir, 'epoch={}.pth.tar'.format(save_epoch))
        
        print('Epoch {} Training Starts'.format(save_epoch))
        
        train_loss, train_acc = train(save_epoch, model, train_batch_num, train_loader, optimizer, args, train_begin)        
        
        
        
        print('Epoch %d (Training) Loss %0.4f Acc %0.2f' % (save_epoch, train_loss, train_acc))
                
        print('Epoch {} Validation Starts'.format(save_epoch))        
        valid_loss, valid_acc = evaluate(model, valid_loader, args)        
        print('Epoch %d (Validation) Loss %0.4f Acc %0.4f' % (save_epoch, valid_loss, valid_acc))
                
        for g in optimizer.param_groups:
            g['lr'] = g['lr'] / args['anneal']
        print('Learning rate annealed to: {lr:.6f}'.format(lr=g['lr']))
        
        args['lr'] = g['lr']
        
        state = {
            'epoch': save_epoch,
            'Model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr': args['lr']
        }
        torch.save(state, save_model_name)
        print('model saved finished at {}'.format(save_model_name))
                    
        summary.add_scalar('train_loss', train_loss, save_epoch)
        summary.add_scalar('valid_loss', valid_loss, save_epoch)
        
        with open(os.path.join(save_dir, 'valid_acc_results.txt'), 'a') as f:
            for_write = 'epoch = {}, valid_acc = {}\n'.format(save_epoch, valid_acc)
            f.write(for_write)
        
        print('Shuffling batches...')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
       

def train(epoch, model, total_batch_size, data_loader, optimizer, args, train_begin):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    accs = AverageMeter('Acc', ':.3e')
        
    progress = ProgressMeter(total_batch_size, [batch_time, data_time, losses, accs], prefix="Epoch: [{}]".format(epoch))
        
    model.train()

    print('train() start')
    
    end = time.time()
    begin = epoch_begin = time.time()

    for i, data in enumerate(data_loader):

        data_time.update(time.time() - end)        
        feats = data
        
        if feats.dim() == 3:
            feats = feats.unsqueeze(0)
            scripts = scripts.unsqueeze(0)
        '''
        if args['aug'] == 1:        
            feats = feats.squeeze(1) # B, D, T
            new_feats = feats.numpy()
            LB = spec_augment(new_feats, frequency_mask_num=1, time_mask_num=1)
            LD = spec_augment(new_feats, frequency_mask_num=2, time_mask_num=2)
            gathered = np.concatenate((new_feats, LB, LD), axis=0)
            feats = torch.from_numpy(gathered)
            feats = feats.unsqueeze(1)        
            scripts = torch.cat([scripts,scripts,scripts], dim=0)
            feat_lengths = torch.cat([feat_lengths,feat_lengths,feat_lengths], dim=0)
        '''        
       
        optimizer.zero_grad()
        
        feats = feats.cuda(args['device'])
        
        output, _ = model(feats)
        loss, acc = model.cpc_loss(feats, steps_predicted=12, n_false_negatives=128, negatives_from_same_seq_only=True, eval_acc=True)   
        
        accs.update(acc, feats.size(0))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 400)
        optimizer.step()
        
        losses.update(loss.item(), feats.size(0))
                                
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args['print_freq'] == 0:
            current = time.time()
            elapsed = current - begin
            epoch_elapsed = (current - epoch_begin) / 60.0
            train_elapsed = (current - train_begin) / 3600.0
            progress.display(i)
            print('elapsed: {:.2f}s {:.2f}m {:.2f}h'.format(elapsed, epoch_elapsed, train_elapsed))
            begin = time.time()
        
        total_steps = ((epoch-1) * total_batch_size) + i + 1
        global summary
                    
        summary.add_scalar('train_loss_steps', losses.avg, total_steps)
        summary.add_scalar('train_acc_steps', accs.avg, total_steps)

    print('train() completed')
    return losses.avg, accs.avg




def evaluate(model, data_loader, args):
    
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    accs = AverageMeter('Acc', ':.3e')
        
    progress = ProgressMeter(len(data_loader), [batch_time, losses, accs], prefix="Evaluation: ")
        
    model.eval()
    
    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(data_loader):
            
            feats = data
        
            if feats.dim() == 3:
                feats = feats.unsqueeze(0)
            
            feats = feats.cuda(args['device'])        
            #output, _ = model(feats)
            loss, acc = model.cpc_loss(feats, steps_predicted=12, n_false_negatives=128, negatives_from_same_seq_only=True, eval_acc=True)
                                    
            losses.update(loss.item(), feats.size(0))
            accs.update(acc, feats.size(0))
    
    print('Loss {loss.avg:.3f} Acc {acc.avg:.3f}'.format(loss=losses, acc=accs))
    return losses.avg, accs.avg



if __name__ == '__main__':
    main()
