# README
This is the code of SFace based on `PyTorch`, we also have a [MxNet](https://github.com/zhongyy/SFace/tree/main/SFace_mxnet) version. 

## Usage Instructions

The code is adopted from [InsightFace](https://github.com/deepinsight/insightface) and [face.evoLVe](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch). I sincerely appreciate for their contributions.

1. Install Pytorch with GPU support (Python 3.6).

2. download the code.

3. The training databases, CASIA-WebFace, VGGFace2 and MS1MV2, evaluation datasets can be downloaded from Data Zoo of [InsightFace](https://github.com/deepinsight/insightface).  

## Train
Run the code to train a model.

(1) Train ResNet50, CASIA-WebFace, ArcFace.
```
CUDA_VISIBLE_DEVICES='0,1' python3 -u train_softmax.py --workers_id 0,1 --batch_size 256 --lr 0.1 --stages 50,70,80 --data_mode casia --net IR_50 --head ArcFace --outdir ./results/IR_50-arc-casia 2>&1|tee ./logs/IR_50-arc-casia.log

```
(2) Train ResNet50, CASIA-WebFace, SFace.
```
CUDA_VISIBLE_DEVICES='0,1' python3 -u train_SFace_torch.py --workers_id 0,1 --batch_size 256 --lr 0.1 --stages 50,70,80 --data_mode casia --net IR_50 --outdir ./results/IR_50-sface-casia --param_a 0.87 --param_b 1.2 2>&1|tee ./logs/IR_50-sfacce-casia.log

```
## Test
Please refer to [InsightFace](https://github.com/deepinsight/insightface) for evaluation on MegaFace and IJB-C.

