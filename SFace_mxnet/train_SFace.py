from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import random
import logging
import pickle
import numpy as np
from image_iter import FaceImageIter
import mxnet as mx
from mxnet import ndarray as nd
import argparse
import mxnet.optimizer as optimizer
sys.path.append(os.path.join(os.path.dirname(__file__), 'common'))
import face_image
sys.path.append(os.path.join(os.path.dirname(__file__), 'eval'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'symbols'))
import fresnet
import verification
import sklearn

os.environ['CUDA_VISIBLE_DEVICES']='0,1'

logger = logging.getLogger()
logger.setLevel(logging.INFO)


args = None


class AccMetric(mx.metric.EvalMetric):
  def __init__(self):
    self.axis = 1
    super(AccMetric, self).__init__(
        'acc', axis=self.axis,
        output_names=None, label_names=None)
    self.losses = []
    self.count = 0

  def update(self, labels, preds):
    self.count+=1
    preds = [preds[1]] #use softmax output
    for label, pred_label in zip(labels, preds):
        if pred_label.shape != label.shape:
            pred_label = mx.ndarray.argmax(pred_label, axis=self.axis)
        pred_label = pred_label.asnumpy().astype('int32').flatten()
        label = label.asnumpy()
        if label.ndim==2:
          label = label[:,0]
        label = label.astype('int32').flatten()
        assert label.shape==pred_label.shape
        self.sum_metric += (pred_label.flat == label.flat).sum()
        self.num_inst += len(pred_label.flat)

class WyiX(mx.metric.EvalMetric):
    def __init__(self):
        self.axis = 1
        super(WyiX, self).__init__(
            name='WyiX', axis=self.axis,
            output_names=None, label_names=None)
        self.num_batch = 0
    def update(self, labels, preds):
        self.num_batch +=1
        loss = preds[-3].asnumpy()[0]
        self.sum_metric += loss
        self.num_inst += 1

class WjX(mx.metric.EvalMetric):
    def __init__(self):
        self.axis = 1
        super(WjX, self).__init__(
            name='WjX', axis=self.axis,
            output_names=None, label_names=None)
        self.num_batch = 0
    def update(self, labels, preds):
        self.num_batch +=1
        loss = preds[-4].asnumpy()[0]
        self.sum_metric += loss
        self.num_inst += 1

class interloss(mx.metric.EvalMetric):
    def __init__(self):
        self.axis = 1
        super(interloss, self).__init__(
            name='inter', axis=self.axis,
            output_names=None, label_names=None)
        self.num_batch = 0
    def update(self, labels, preds):
        self.num_batch +=1
        loss = preds[-2].asnumpy()[0]
        self.sum_metric += loss
        self.num_inst += 1

class intraloss(mx.metric.EvalMetric):
    def __init__(self):
        self.axis = 1
        super(intraloss, self).__init__(
            name='intra', axis=self.axis,
            output_names=None, label_names=None)
        self.num_batch = 0
    def update(self, labels, preds):
        self.num_batch +=1
        loss = preds[-1].asnumpy()[0]
        self.sum_metric += loss
        self.num_inst += 1

def parse_args():
  parser = argparse.ArgumentParser(description='Train face network')
  # general
  parser.add_argument('--data-dir', default='', help='training set directory')
  parser.add_argument('--eval-dir', default='', help='evaluation set directory')
  parser.add_argument('--prefix', default='../model/model', help='directory to save model.')
  parser.add_argument('--pretrained', default='', help='pretrained model to load')
  parser.add_argument('--ckpt', type=int, default=1, help='checkpoint saving option. 0: discard saving. 1: save when necessary. 2: always save')
  parser.add_argument('--loss-type', type=int, default=4, help='loss type')
  parser.add_argument('--verbose', type=int, default=2000, help='do verification testing and model saving every verbose batches')
  parser.add_argument('--max-steps', type=int, default=0, help='max training batches')
  parser.add_argument('--end-epoch', type=int, default=100000, help='training epoch size.')
  parser.add_argument('--network', default='r50', help='specify network')
  parser.add_argument('--version-se', type=int, default=0, help='whether to use se in network')
  parser.add_argument('--version-input', type=int, default=1, help='network input config')
  parser.add_argument('--version-output', type=str, default='E', help='network embedding output config')
  parser.add_argument('--version-unit', type=int, default=3, help='resnet unit config')
  parser.add_argument('--version-act', type=str, default='prelu', help='network activation config')
  parser.add_argument('--use-deformable', type=int, default=0, help='use deformable cnn in network')
  parser.add_argument('--lr', type=float, default=0.1, help='start learning rate')
  parser.add_argument('--lr-steps', type=str, default='', help='steps of lr changing')
  parser.add_argument('--wd', type=float, default=0.0005, help='weight decay')
  parser.add_argument('--fc7-wd-mult', type=float, default=1.0, help='weight decay mult for fc7')
  parser.add_argument('--bn-mom', type=float, default=0.9, help='bn mom')
  parser.add_argument('--mom', type=float, default=0.9, help='momentum')
  parser.add_argument('--emb-size', type=int, default=512, help='embedding length')
  parser.add_argument('--per-batch-size', type=int, default=100, help='batch size in each context')
  parser.add_argument('--margin-s', type=float, default=64.0, help='scale for feature')
  parser.add_argument('--param-a', type=float, default=80.0, help='margin for loss')
  parser.add_argument('--param-b', type=float, default=0.87, help='scale for feature')
  parser.add_argument('--param-c', type=float, default=80.0, help='scale for feature')
  parser.add_argument('--param-d', type=float, default=1.2, help='scale for feature')
  parser.add_argument('--rand-mirror', type=int, default=1, help='if do random mirror in training')
  parser.add_argument('--cutoff', type=int, default=0, help='cut off aug')
  parser.add_argument('--target', type=str, default='lfw,cfp_fp,agedb_30', help='verification targets')
  parser.add_argument('--log-file', type=str, default='trainlog', help='the name of log file')
  parser.add_argument('--log-dir', type=str, default='/home/zhongyaoyao/insightface/', help='directory of the log file')
  args = parser.parse_args()
  return args


def get_symbol(args, arg_params, aux_params):
  print('init resnet', args.num_layers)
  embedding = fresnet.get_symbol(args.emb_size, args.num_layers,
        version_se=args.version_se, version_input=args.version_input,
        version_output=args.version_output, version_unit=args.version_unit,
        version_act=args.version_act)
  all_label = mx.symbol.Variable('softmax_label')
  gt_label = all_label
  _weight = mx.symbol.Variable("fc7_weight", shape=(args.num_classes, args.emb_size), lr_mult=1.0, wd_mult=args.fc7_wd_mult)
  s = args.margin_s
  aa = args.param_a
  bb = args.param_b
  cc = args.param_c
  dd = args.param_d
  assert s > 0.0
  _weight = mx.symbol.L2Normalization(_weight, mode='instance')
  nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n') * s

  fc7 = mx.sym.FullyConnected(data=nembedding, weight=_weight, no_bias=True, num_hidden=args.num_classes, name='fc7')
  gt_one_hot = mx.sym.one_hot(gt_label, depth=args.num_classes, on_value=1.0, off_value=0.0)
  gt_j_hot = mx.sym.one_hot(gt_label, depth=args.num_classes, on_value=0.0, off_value=1.0)
  WyiX = mx.sym.sum(mx.sym.broadcast_mul(gt_one_hot, fc7), axis=1)
  theta_yi = mx.symbol.BlockGrad(mx.sym.arccos(mx.symbol.BlockGrad(WyiX/s)))
  W_intra = 1/(mx.symbol.BlockGrad(1+mx.symbol.BlockGrad(mx.sym.exp(-(aa*(mx.symbol.BlockGrad(theta_yi)-bb))))))
  en_intra = - W_intra * WyiX
  en_intra = mx.sym.mean(en_intra)
  en_intra = mx.sym.MakeLoss(en_intra)

  WjX = mx.sym.broadcast_mul(gt_j_hot, fc7)
  theta_j = mx.symbol.BlockGrad(mx.sym.arccos(mx.symbol.BlockGrad(WjX/s)))
  W_inter = 1/(mx.symbol.BlockGrad(1+mx.symbol.BlockGrad(mx.sym.exp(cc*(mx.symbol.BlockGrad(theta_j)-dd)))))
  en_inter = mx.sym.BlockGrad(W_inter)*WjX
  en_inter = mx.sym.sum(en_inter, axis=1)
  en_inter = mx.sym.mean(en_inter)
  en_inter = mx.sym.MakeLoss(en_inter)

  out_list = [mx.symbol.BlockGrad(embedding)]
  out_list.append(
      mx.symbol.BlockGrad(mx.symbol.Softmax(data=fc7, label=gt_label, name='softmax2', normalization='valid')))
  out_list.append(mx.symbol.BlockGrad(mx.sym.mean(mx.symbol.BlockGrad(WjX/s))))
  out_list.append(mx.symbol.BlockGrad(mx.sym.mean(mx.symbol.BlockGrad(WyiX/s))))
  out_list.append(en_inter)
  out_list.append(en_intra)
  out = mx.symbol.Group(out_list)
  return (out, arg_params, aux_params)

def train_net(args):
    ctx = []
    cvd = os.environ['CUDA_VISIBLE_DEVICES'].strip()
    if len(cvd)>0:
      for i in xrange(len(cvd.split(','))):
        ctx.append(mx.gpu(i))
    if len(ctx)==0:
      ctx = [mx.cpu()]
      print('use cpu')
    else:
      print('gpu num:', len(ctx))
    prefix = args.prefix
    prefix_dir = os.path.dirname(prefix)
    if not os.path.exists(prefix_dir):
      os.makedirs(prefix_dir)
    end_epoch = args.end_epoch
    args.ctx_num = len(ctx)
    args.num_layers = int(args.network[1:])
    print('num_layers', args.num_layers)
    if args.per_batch_size==0:
      args.per_batch_size = 128
    args.batch_size = args.per_batch_size*args.ctx_num
    args.rescale_threshold = 0
    args.image_channel = 3

    data_dir_list = args.data_dir.split(',')
    assert len(data_dir_list)==1
    data_dir = data_dir_list[0]
    prop = face_image.load_property(data_dir)
    args.num_classes = prop.num_classes
    image_size = prop.image_size
    args.image_h = image_size[0]
    args.image_w = image_size[1]
    print('image_size', image_size)
    assert(args.num_classes>0)
    print('num_classes', args.num_classes)
    path_imgrec = os.path.join(data_dir, "train.rec")


    print('Called with argument:', args)
    data_shape = (args.image_channel,image_size[0],image_size[1])
    mean = None
    begin_epoch = 0
    base_lr = args.lr
    base_wd = args.wd
    base_mom = args.mom
    if len(args.pretrained)==0:
      arg_params = None
      aux_params = None
      sym, arg_params, aux_params = get_symbol(args, arg_params, aux_params)
    else:
      vec = args.pretrained.split(',')
      print('loading', vec)
      _, arg_params, aux_params = mx.model.load_checkpoint(vec[0], int(vec[1]))
      sym, arg_params, aux_params = get_symbol(args, arg_params, aux_params)

    model = mx.mod.Module(
        context       = ctx,
        symbol        = sym,
    )
    val_dataiter = None

    train_dataiter = FaceImageIter(
        batch_size           = args.batch_size,
        data_shape           = data_shape,
        path_imgrec          = path_imgrec,
        shuffle              = True,
        rand_mirror          = args.rand_mirror,
        mean                 = mean,
        cutoff               = args.cutoff,
    )

    eval_metrics = [mx.metric.create(AccMetric()),
                    mx.metric.create(interloss()), mx.metric.create(intraloss()),
                    mx.metric.create(WyiX()), mx.metric.create(WjX())]

    if args.network[0]=='r' or args.network[0]=='y':
      initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="out", magnitude=2) #resnet style
    elif args.network[0]=='i' or args.network[0]=='x':
      initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2) #inception
    else:
      initializer = mx.init.Xavier(rnd_type='uniform', factor_type="in", magnitude=2)
    _rescale = 1.0/args.ctx_num
    opt = optimizer.SGD(learning_rate=base_lr, momentum=base_mom, wd=base_wd, rescale_grad=_rescale)
    som = 20
    _cb = mx.callback.Speedometer(args.batch_size, som)

    ver_list = []
    ver_name_list = []
    for name in args.target.split(','):
      #path = os.path.join(data_dir, name + ".bin")
      path = os.path.join(args.eval_dir,name+".bin")
      if os.path.exists(path):
        data_set = verification.load_bin(path, image_size)
        ver_list.append(data_set)
        ver_name_list.append(name)
        print('ver', name)



    def ver_test(nbatch):
      results = []
      for i in xrange(len(ver_list)):
        acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test(ver_list[i], model, args.batch_size, 10, None, None)
        print('[%s][%d]XNorm: %f' % (ver_name_list[i], nbatch, xnorm))
        print('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (ver_name_list[i], nbatch, acc2, std2))
        results.append(acc2)
      return results



    highest_acc = [0.0, 0.0 , 0.0 , 0.0 , 0.0, 0.0]  #evaluation
    global_step = [0]
    save_step = [0]
    if len(args.lr_steps)==0:
      lr_steps = [16000, 24000]
      if args.loss_type>=1 and args.loss_type<=7:
        lr_steps = [100000, 140000, 160000]
      p = 512.0/args.batch_size
      for l in xrange(len(lr_steps)):
        lr_steps[l] = int(lr_steps[l]*p)
    else:
      lr_steps = [int(x) for x in args.lr_steps.split(',')]
    print('lr_steps', lr_steps)
    def _batch_callback(param):
      #global global_step
      global_step[0]+=1
      mbatch = global_step[0]
      for _lr in lr_steps:
        if mbatch==_lr:
          opt.lr *= 0.1
          print('lr change to', opt.lr)
          break

      _cb(param)

      if mbatch%1000==0:
        print('lr-batch-epoch:',opt.lr,param.nbatch,param.epoch)
      if mbatch==1:
        ver_test(mbatch)
      if mbatch>=0 and mbatch%args.verbose==0:
        arg, aux = model.get_params()
        model.set_params(arg, aux)
        mx.model.save_checkpoint(prefix, 0, model.symbol, arg, aux)
        acc_list = ver_test(mbatch)
        save_step[0]+=1
        msave = save_step[0]
        do_save = False
        sum_acc = 0
        if mbatch % 10000 == 0:
            do_save = True
        for i in xrange(len(acc_list)):
            sum_acc += acc_list[i]
        print('[%d]sum_acc: %1.5f' % (mbatch, sum_acc / 5.0))
        if sum_acc > 0.933:
            do_save = True
        if sum_acc > highest_acc[-1]:
            highest_acc[-1] = sum_acc
            if acc_list[0]>=0.99:
                do_save = True
        if len(acc_list)>0:
          lfw_score = acc_list[0]
          if lfw_score>highest_acc[0]:
            highest_acc[0] = lfw_score
            if lfw_score>=0.99:
              do_save = True
          if acc_list[1]>=highest_acc[1]:
            highest_acc[1] = acc_list[1]
            if lfw_score>=0.99:
              do_save = True
          if acc_list[2]>=highest_acc[2]:
            highest_acc[2] = acc_list[2]
            if lfw_score>=0.99:
              do_save = True
          if acc_list[3]>=highest_acc[3]:
            highest_acc[3] = acc_list[3]
            if lfw_score>=0.99:
              do_save = True
          if acc_list[4]>=highest_acc[4]:
            highest_acc[4] = acc_list[4]
            if lfw_score>=0.99:
              do_save = True

        if args.ckpt==0:
          do_save = False
        elif args.ckpt>1:
          do_save = True
        if do_save:
          print('saving', msave)
          arg, aux = model.get_params()
          mx.model.save_checkpoint(prefix, msave, model.symbol, arg, aux)
        print('[%d]Accuracy-Highest: %1.5f %1.5f %1.5f %1.5f %1.5f %1.5f' % (
        mbatch, highest_acc[-1] / 5.0, highest_acc[0], highest_acc[1], highest_acc[2], highest_acc[3], highest_acc[4]))
      if args.max_steps>0 and mbatch>args.max_steps:
        sys.exit(0)

    epoch_cb = None

    model.fit(train_dataiter,
        begin_epoch        = begin_epoch,
        num_epoch          = end_epoch,
        eval_data          = val_dataiter,
        eval_metric        = eval_metrics,
        kvstore            = 'local',
        optimizer          = opt,
        #optimizer_params   = optimizer_params,
        initializer        = initializer,
        arg_params         = arg_params,
        aux_params         = aux_params,
        allow_missing      = True,
        batch_end_callback = _batch_callback,
        epoch_end_callback = epoch_cb )

def main():
    global args
    args = parse_args()
    train_net(args)

if __name__ == '__main__':
    main()

