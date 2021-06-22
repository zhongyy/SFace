import torch, os
import yaml
from IPython import embed


def get_yaml_data(yaml_file):
    file = open(yaml_file, 'r', encoding="utf-8")
    file_data = file.read()
    file.close()
    data = yaml.load(file_data)
    return data

def get_config(args):
    configuration = dict(
        SEED=1337,  # random seed for reproduce results
        INPUT_SIZE=[112, 112],
        EMBEDDING_SIZE=512, # embedding size
        DROP_LAST=True,
        WEIGHT_DECAY=5e-4,
        MOMENTUM=0.9,
    )

    if args.workers_id == 'cpu' or not torch.cuda.is_available():
        configuration['GPU_ID'] = []
        print("check", args.workers_id, torch.cuda.is_available())
    else:
        configuration['GPU_ID'] = [int(i) for i in args.workers_id.split(',')]
    if len(configuration['GPU_ID']) == 0:
        configuration['DEVICE'] = torch.device('cpu')
        configuration['MULTI_GPU'] = False
    else:
        configuration['DEVICE'] = torch.device('cuda:%d' % configuration['GPU_ID'][0])
        if len(configuration['GPU_ID']) == 1:
            configuration['MULTI_GPU'] = False
        else:
            configuration['MULTI_GPU'] = True

    configuration['NUM_EPOCH'] = args.epochs
    configuration['STAGES'] = [int(i) for i in args.stages.split(',')]
    configuration['LR'] = args.lr
    configuration['BATCH_SIZE'] = args.batch_size

    if args.data_mode == 'casia':
        configuration['DATA_ROOT'] = '../datasets/faces_webface_112x112/' # the dir for training
    else:
        raise Exception(args.data_mode)

    configuration['EVAL_PATH'] = '../eval/' # the dir for validation

    assert args.net in ['IR_50', 'IR_101', 'MobileFaceNet']


    configuration['BACKBONE_NAME'] = args.net
    assert args.head in ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax','SFaceLoss']
    configuration['HEAD_NAME'] = args.head

    configuration['TARGET'] = [i for i in args.target.split(',')]

    if args.resume_backbone:
        configuration['BACKBONE_RESUME_ROOT'] = args.resume_backbone  # the dir to resume training from a saved checkpoint
        configuration['HEAD_RESUME_ROOT'] = args.resume_head  # the dir to resume training from a saved checkpoint
    else:
        configuration['BACKBONE_RESUME_ROOT'] = ''
        configuration['HEAD_RESUME_ROOT'] = ''
    configuration['WORK_PATH'] = args.outdir  # the dir to save your checkpoints
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    return configuration
