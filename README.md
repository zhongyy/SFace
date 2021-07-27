# SFace
Code of TIP2021 Paper [《SFace: Sigmoid-Constrained Hypersphere Loss for Robust Face Recognition》](https://ieeexplore.ieee.org/document/9318547). 

We provide both [MxNet](https://github.com/zhongyy/SFace/tree/main/SFace_mxnet), [PyTorch](https://github.com/zhongyy/SFace/tree/main/SFace_torch) and [Jittor](https://github.com/liubingyuu/jittorface) versions.

## Abstract 
Deep face recognition has achieved great success due to large-scale training databases and rapidly developing loss functions. The existing algorithms devote to realizing an ideal idea: minimizing the intra-class distance and maximizing the inter-class distance. However, they may neglect that there are also low quality training images which should not be optimized in this strict way. Considering the imperfection of training databases, we propose that intra-class and inter-class objectives can be optimized in a moderate way to mitigate overfitting problem, and further propose a novel loss function, named sigmoid-constrained hypersphere loss (SFace). Specifically, SFace imposes intra-class and inter-class constraints on a hypersphere manifold, which are controlled by two sigmoid gradient re-scale functions respectively. The sigmoid curves precisely re-scale the intra-class and inter-class gradients so that training samples can be optimized to some degree. Therefore, SFace can make a better balance between decreasing the intra-class distances for clean examples and preventing overfitting to the label noise, and contributes more robust deep face recognition models. Extensive experiments of models trained on CASIA-WebFace, VGGFace2, and MS-Celeb-1M databases, and evaluated on several face recognition benchmarks, such as LFW, MegaFace and IJB-C databases, have demonstrated the superiority of SFace. 

![arch](https://github.com/zhongyy/SFace/blob/main/a.jpg)

## Usage
Please check your familiar version: [MxNet](https://github.com/zhongyy/SFace/tree/main/SFace_mxnet) and [PyTorch](https://github.com/zhongyy/SFace/tree/main/SFace_torch), [PyTorch](https://github.com/zhongyy/SFace/tree/main/SFace_torch) and [Jittor](https://github.com/liubingyuu/jittorface) versions.
