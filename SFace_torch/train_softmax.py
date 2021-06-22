import os, argparse, sklearn
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from config import get_config
from image_iter_rec import FaceDataset
from backbone.model_irse import IR_50, IR_101
from backbone.model_mobilefacenet import MobileFaceNet
from head.metrics import Softmax, ArcFace, CosFace, SphereFace, Am_softmax
from util.utils import separate_irse_bn_paras, separate_resnet_bn_paras, separate_mobilefacenet_bn_paras
from util.utils import get_val_data, perform_val, get_time, buffer_val, AverageMeter, train_accuracy

import math
import time

def xavier_normal_(tensor, gain=1., mode='avg'):
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    if mode == 'avg':
        fan = fan_in + fan_out
    elif mode == 'in':
        fan = fan_in
    elif mode == 'out':
        fan = fan_out
    else:
        raise Exception('wrong mode')
    std = gain * math.sqrt(2.0 / float(fan))

    return nn.init._no_grad_normal_(tensor, 0., std)


def weight_init(m):
    #print(m)
    if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.fill_(1)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.zero_()
        if hasattr(m, 'running_mean') and m.running_mean is not None:
            m.running_mean.data.zero_()
        if hasattr(m, 'running_var') and m.running_var is not None:
            m.running_var.data.fill_(1)
    elif isinstance(m, nn.PReLU):
        m.weight.data.fill_(1)
    else:
        if hasattr(m, 'weight') and m.weight is not None:
            xavier_normal_(m.weight.data, gain=2, mode='out')
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.zero_()

def schedule_lr(optimizer):
    for params in optimizer.param_groups:
        params['lr'] /= 10.

    print(optimizer)


def need_save(acc, highest_acc):
    do_save = False
    save_cnt = 0
    if acc[0] > 0.98:
        do_save = True
    for i, accuracy in enumerate(acc):
        if accuracy > highest_acc[i]:
            highest_acc[i] = accuracy
            do_save = True
        if i > 0 and accuracy >= highest_acc[i]-0.002:
            save_cnt += 1
    if save_cnt >= len(acc)*3/4 and acc[0]>0.99:
        do_save = True
    print("highest_acc:", highest_acc)
    return do_save

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument('--workers_id', help="gpu ids or cpu", default='cpu', type=str)
    parser.add_argument('--epochs', help="training epochs", default=125, type=int)
    parser.add_argument('--stages', help="training stages", default='35,65,95', type=str)
    parser.add_argument('--lr',help='learning rate',default=1e-1, type=float)
    parser.add_argument('--batch_size', help="batch_size", default=256, type=int)
    parser.add_argument('--data_mode', help="use which database, [casia, vgg, ms1m, retina, ms1mr]",default='ms1m', type=str)
    parser.add_argument('--net', help="which network, ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152', 'MobileFaceNet','HRNet']",default='MobileFaceNet', type=str)
    parser.add_argument('--head', help="head type, ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']", default='ArcFace', type=str)
    parser.add_argument('--target', help="verification targets", default='lfw,talfw,sllfw,calfw,cplfw,cfp_fp,agedb_30', type=str)
    parser.add_argument('--resume_backbone', help="resume backbone model", default='', type=str)
    parser.add_argument('--resume_head', help="resume head model", default='', type=str)
    parser.add_argument('--outdir', help="output dir", default='test_dir', type=str)
    args = parser.parse_args()

    #======= hyperparameters & data loaders =======#
    cfg = get_config(args)

    SEED = cfg['SEED'] # random seed for reproduce results
    torch.manual_seed(SEED)

    DATA_ROOT = cfg['DATA_ROOT']  # the parent root where your train data are stored
    EVAL_PATH = cfg['EVAL_PATH']  # the parent root where your val data are stored
    WORK_PATH = cfg['WORK_PATH']  # the root to buffer your checkpoints and to log your train/val status
    BACKBONE_RESUME_ROOT = cfg['BACKBONE_RESUME_ROOT']  # the root to resume training from a saved checkpoint
    HEAD_RESUME_ROOT = cfg['HEAD_RESUME_ROOT']  # the root to resume training from a saved checkpoint

    BACKBONE_NAME = cfg['BACKBONE_NAME'] # support: ['IR_50', 'IR_101']
    HEAD_NAME = cfg['HEAD_NAME'] # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']


    INPUT_SIZE = cfg['INPUT_SIZE']
    EMBEDDING_SIZE = cfg['EMBEDDING_SIZE'] # feature dimension
    BATCH_SIZE = cfg['BATCH_SIZE']
    DROP_LAST = cfg['DROP_LAST'] # whether drop the last batch to ensure consistent batch_norm statistics
    LR = cfg['LR'] # initial LR
    NUM_EPOCH = cfg['NUM_EPOCH']
    WEIGHT_DECAY = cfg['WEIGHT_DECAY']
    MOMENTUM = cfg['MOMENTUM']
    STAGES = cfg['STAGES'] # epoch stages to decay learning rate

    DEVICE = cfg['DEVICE']
    MULTI_GPU = cfg['MULTI_GPU'] # flag to use multiple GPUs
    GPU_ID = cfg['GPU_ID'] # specify your GPU ids
    print('GPU_ID', GPU_ID)
    TARGET = cfg['TARGET']
    print("=" * 60)
    print("Overall Configurations:")
    print(cfg)
    with open(os.path.join(WORK_PATH, 'config.txt'), 'w') as f:
        f.write(str(cfg))
    print("=" * 60)

    writer = SummaryWriter(WORK_PATH) # writer for buffering intermedium results
    torch.backends.cudnn.benchmark = True

    with open(os.path.join(DATA_ROOT, 'property'), 'r') as f:
        NUM_CLASS, h, w = [int(i) for i in f.read().split(',')]
    assert h == INPUT_SIZE[0] and w == INPUT_SIZE[1]

    dataset = FaceDataset(os.path.join(DATA_ROOT, 'train.rec'), rand_mirror=True)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=len(GPU_ID), drop_last=True)

    print("Number of Training Classes: {}".format(NUM_CLASS))

    vers = get_val_data(EVAL_PATH, TARGET)
    highest_acc = [0.0 for t in TARGET]


    #======= model & loss & optimizer =======#
    BACKBONE_DICT = {'IR_50': IR_50(INPUT_SIZE),
                     'IR_101': IR_101(INPUT_SIZE),
                     'MobileFaceNet': MobileFaceNet(EMBEDDING_SIZE)}
    BACKBONE = BACKBONE_DICT[BACKBONE_NAME]
    print("=" * 60)
    print(BACKBONE)
    print("{} Backbone Generated".format(BACKBONE_NAME))
    print("=" * 60)

    HEAD_DICT = {'Softmax': Softmax(in_features = EMBEDDING_SIZE, out_features = NUM_CLASS, device_id = GPU_ID),
                 'ArcFace': ArcFace(in_features = EMBEDDING_SIZE, out_features = NUM_CLASS, device_id = GPU_ID),
                 'CosFace': CosFace(in_features = EMBEDDING_SIZE, out_features = NUM_CLASS, device_id = GPU_ID),
                 'SphereFace': SphereFace(in_features = EMBEDDING_SIZE, out_features = NUM_CLASS, device_id = GPU_ID),
                 'Am_softmax': Am_softmax(in_features = EMBEDDING_SIZE, out_features = NUM_CLASS, device_id = GPU_ID)}
    HEAD = HEAD_DICT[HEAD_NAME]
    print("=" * 60)
    print(HEAD)
    print("{} Head Generated".format(HEAD_NAME))
    print("=" * 60)

    LOSS = nn.CrossEntropyLoss()

    if BACKBONE_NAME.find("IR") >= 0:
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_irse_bn_paras(BACKBONE) # separate batch_norm parameters from others; do not do weight decay for batch_norm parameters to improve the generalizability
        _, head_paras_wo_bn = separate_irse_bn_paras(HEAD)
    elif BACKBONE_NAME.find("MobileFace") >= 0:
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_mobilefacenet_bn_paras(BACKBONE) # separate batch_norm parameters from others; do not do weight decay for batch_norm parameters to improve the generalizability
        _, head_paras_wo_bn = separate_mobilefacenet_bn_paras(HEAD)
    else:
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_resnet_bn_paras(BACKBONE) # separate batch_norm parameters from others; do not do weight decay for batch_norm parameters to improve the generalizability
        _, head_paras_wo_bn = separate_resnet_bn_paras(HEAD)
    OPTIMIZER = optim.SGD([{'params': backbone_paras_wo_bn + head_paras_wo_bn, 'weight_decay': WEIGHT_DECAY}, {'params': backbone_paras_only_bn}], lr = LR, momentum = MOMENTUM)
    print("=" * 60)
    print(OPTIMIZER)
    print("Optimizer Generated")
    print("=" * 60)

    BACKBONE.apply(weight_init)
    HEAD.apply(weight_init)

    # optionally resume from a checkpoint
    if BACKBONE_RESUME_ROOT and HEAD_RESUME_ROOT:
        print("=" * 60)
        print(BACKBONE_RESUME_ROOT,HEAD_RESUME_ROOT)
        if os.path.isfile(BACKBONE_RESUME_ROOT) and os.path.isfile(HEAD_RESUME_ROOT):
            print("Loading Backbone Checkpoint '{}'".format(BACKBONE_RESUME_ROOT))
            BACKBONE.load_state_dict(torch.load(BACKBONE_RESUME_ROOT))
            print("Loading Head Checkpoint '{}'".format(HEAD_RESUME_ROOT))
            HEAD.load_state_dict(torch.load(HEAD_RESUME_ROOT))
        else:
            print("No Checkpoint Found at '{}' and '{}'. Please Have a Check or Continue to Train from Scratch".format(BACKBONE_RESUME_ROOT, HEAD_RESUME_ROOT))
        print("=" * 60)

    if MULTI_GPU:
        # multi-GPU setting
        BACKBONE = nn.DataParallel(BACKBONE, device_ids = GPU_ID)
        BACKBONE = BACKBONE.to(DEVICE)
    else:
        # single-GPU setting
        BACKBONE = BACKBONE.to(DEVICE)

    #======= train & validation & save checkpoint =======#
    DISP_FREQ = 20 # frequency to display training loss & acc
    VER_FREQ = 2000
    batch = 0  # batch index

    losses = AverageMeter()
    top1 = AverageMeter()


    BACKBONE.train()  # set to training mode
    HEAD.train()
    for epoch in range(NUM_EPOCH):
        
        if epoch in STAGES:
            schedule_lr(OPTIMIZER)

        last_time = time.time()

        for inputs, labels in iter(trainloader):

            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE).long()
            features = BACKBONE(inputs)
            outputs = HEAD(features, labels)
            loss = LOSS(outputs, labels)

            prec1 = train_accuracy(outputs.data, labels, topk = (1,))

            losses.update(loss.data.item(), inputs.size(0))
            top1.update(prec1.data.item(), inputs.size(0))

            # compute gradient and do SGD step
            OPTIMIZER.zero_grad()
            loss.backward()
            OPTIMIZER.step()
            
            # dispaly training loss & acc every DISP_FREQ (buffer for visualization)
            if ((batch + 1) % DISP_FREQ == 0) and batch != 0:
                epoch_loss = losses.avg
                epoch_acc = top1.avg
                writer.add_scalar("Training/Training_Loss", epoch_loss, batch + 1)
                writer.add_scalar("Training/Training_Accuracy", epoch_acc, batch + 1)

                batch_time = time.time() - last_time
                last_time = time.time()

                print('Epoch {} Batch {}\t'
                      'Speed: {speed:.2f} samples/s\t'
                      'Training Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch + 1, batch + 1, speed=inputs.size(0) * DISP_FREQ / float(batch_time),
                    loss=losses, top1=top1))
                #print("=" * 60)
                losses = AverageMeter()
                top1 = AverageMeter()

            if ((batch + 1) % VER_FREQ == 0) and batch != 0: #perform validation & save checkpoints (buffer for visualization)
                for params in OPTIMIZER.param_groups:
                    lr = params['lr']
                    break
                print("Learning rate %f"%lr)
                print("Perform Evaluation on", TARGET, ", and Save Checkpoints...")
                acc = []
                for ver in vers:
                    name, data_set, issame = ver
                    accuracy, std, xnorm, best_threshold, roc_curve = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, data_set, issame)
                    buffer_val(writer, name, accuracy, std, xnorm, best_threshold, roc_curve, batch + 1)
                    print('[%s][%d]XNorm: %1.5f' % (name, batch+1, xnorm))
                    print('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (name, batch+1, accuracy, std))
                    print('[%s][%d]Best-Threshold: %1.5f' % (name, batch+1, best_threshold))
                    acc.append(accuracy)

                # save checkpoints per epoch
                if need_save(acc, highest_acc):
                    if MULTI_GPU:
                        torch.save(BACKBONE.module.state_dict(), os.path.join(WORK_PATH, "Backbone_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(BACKBONE_NAME, epoch + 1, batch + 1, get_time())))
                        torch.save(HEAD.state_dict(), os.path.join(WORK_PATH, "Head_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(HEAD_NAME, epoch + 1, batch + 1, get_time())))
                    else:
                        torch.save(BACKBONE.state_dict(), os.path.join(WORK_PATH, "Backbone_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(BACKBONE_NAME, epoch + 1, batch + 1, get_time())))
                        torch.save(HEAD.state_dict(), os.path.join(WORK_PATH, "Head_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(HEAD_NAME, epoch + 1, batch + 1, get_time())))
                BACKBONE.train()  # set to training mode

            batch += 1 # batch index

