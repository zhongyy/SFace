CUDA_VISIBLE_DEVICES='0,1' python3 -u train_softmax.py --workers_id 0,1 --batch_size 256 --lr 0.1 --stages 50,70,80 --data_mode casia --net IR_50 --head ArcFace --outdir ./results/IR_50-arc-casia 2>&1|tee ./logs/IR_50-arc-casia.log

CUDA_VISIBLE_DEVICES='0,1' python3 -u train_SFace_torch.py --workers_id 0,1 --batch_size 256 --lr 0.1 --stages 50,70,80 --data_mode casia --net IR_50 --outdir ./results/IR_50-sface-casia --param_a 0.87 --param_b 1.2 2>&1|tee ./logs/IR_50-sfacce-casia.log
