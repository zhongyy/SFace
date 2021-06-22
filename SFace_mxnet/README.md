# README
This is the code of SFace based on `MxNet`, we also have a [PyTorch](https://github.com/zhongyy/SFace/tree/main/SFace_torch) version. 

## Usage Instructions

The code is adopted from [InsightFace](https://github.com/deepinsight/insightface). I sincerely appreciate for their contributions.

1. Install MxNet with GPU support (Python 2.7).

```
pip install mxnet-cu90
```
2. download the code.

3. The training databases, CASIA-WebFace, VGGFace2 and MS1MV2, evaluation datasets can be downloaded from Data Zoo of [InsightFace](https://github.com/deepinsight/insightface).  

## Train
Run the code to train a model.

(1) Train ResNet50, CASIA-WebFace, SFace.
```
CUDA_VISIBLE_DEVICES='0,1' python -u train_SFace.py --network r50 --param-a 0.87 --param-b 1.2 --data-dir ../datasets/faces_webface_112x112/  --lr 0.1 --lr-steps 100000,140000,160000 --end-epoch 100 --verbose 2000 --per-batch-size 128 --prefix ../models/r50_webface_sface/model --target lfw,cplfw,calfw,cfp_fp,agedb_30 2>&1|tee r50_webface_sface.log
```
(2) Train ResNet50, VGGFace2, SFace.
```
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train_SFace.py --network r50 --param-a 0.88 --param-b 1.25 --data-dir ../datasets/faces_vgg_112x112/ --lr 0.1 --lr-steps 100000,160000,220000 --end-epoch 30 --verbose 2000 --per-batch-size 128 --prefix ../models/r100_vgg2_sFace/model --target  lfw,cplfw,calfw,agedb_30 2>&1|tee r100_vgg2_sFace.log
```
(3) Train ResNet100, MS1MV2, SFace.
```
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train_SFace.py --network r100 --param-a 0.9 --param-b 1.2 --data-dir ../datasets/ms_emore/ --lr 0.1 --lr-steps 100000,160000,220000 --end-epoch 30 --verbose 2000 --per-batch-size 128 --prefix ../models/r100_emore_sFace/model --target  lfw,cplfw,calfw,agedb_30 2>&1|tee r100_emore_sFace.log

```
## Test
Please refer to [InsightFace](https://github.com/deepinsight/insightface) for evaluation on MegaFace and IJB-C.

