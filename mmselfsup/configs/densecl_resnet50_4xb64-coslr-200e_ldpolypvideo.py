model = dict(
    type='DenseCL',
    queue_len=65536,
    feat_dim=128,
    momentum=0.999,
    loss_lambda=0.5,
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=[4],
        norm_cfg=dict(type='BN'),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='DenseCLNeck',
        in_channels=2048,
        hid_channels=2048,
        out_channels=128,
        num_grid=None),
    head=dict(type='ContrastiveHead', temperature=0.2))
data_source = 'LDPolypVideo'
dataset_type = 'MultiViewDataset'
img_norm_cfg = dict(mean=[0.654, 0.424, 0.266], std=[0.257, 0.182, 0.148])
train_pipeline = [
    dict(type='RandomResizedCrop', size=224, scale=(0.2, 1.0)),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1)
        ],
        p=0.8),
    dict(type='RandomGrayscale', p=0.2),
    dict(type='GaussianBlur', sigma_min=0.1, sigma_max=2.0, p=0.5),
    dict(type='RandomHorizontalFlip'),
    dict(type='ToTensor'),
    dict(
        type='Normalize',
        mean=[0.654, 0.424, 0.266],
        std=[0.257, 0.182, 0.148])
]
prefetch = False
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
    drop_last=True,
    train=dict(
        type='MultiViewDataset',
        data_source=dict(
            type='LDPolypVideo',
            data_prefix='/home/wsco/gty/datasets/LDPolypVideo/PreTrain',
            ann_file=None),
        num_views=[2],
        pipelines=[[{
            'type': 'RandomResizedCrop',
            'size': 224,
            'scale': (0.2, 1.0)
        }, {
            'type':
            'RandomAppliedTrans',
            'transforms': [{
                'type': 'ColorJitter',
                'brightness': 0.4,
                'contrast': 0.4,
                'saturation': 0.4,
                'hue': 0.1
            }],
            'p':
            0.8
        }, {
            'type': 'RandomGrayscale',
            'p': 0.2
        }, {
            'type': 'GaussianBlur',
            'sigma_min': 0.1,
            'sigma_max': 2.0,
            'p': 0.5
        }, {
            'type': 'RandomHorizontalFlip'
        }, {
            'type': 'ToTensor'
        }, {
            'type': 'Normalize',
            'mean': [0.654, 0.424, 0.266],
            'std': [0.257, 0.182, 0.148]
        }]],
        prefetch=False))
optimizer = dict(type='SGD', lr=0.03, weight_decay=0.0001, momentum=0.9)
optimizer_config = dict()
lr_config = dict(policy='CosineAnnealing', min_lr=0.0)
runner = dict(type='EpochBasedRunner', max_epochs=200)
checkpoint_config = dict(interval=10, max_keep_ckpts=3)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
cudnn_benchmark = True
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
persistent_workers = True
opencv_num_threads = 0
mp_start_method = 'fork'
work_dir = './work_dirs/selfsup/densecl_resnet50_4xb64-coslr-200e_ldpolypvideo'
auto_resume = False
gpu_ids = range(0, 4)
