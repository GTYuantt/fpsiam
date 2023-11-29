model = dict(
    type='SimCLR',
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=[4],
        norm_cfg=dict(type='SyncBN'),
        zero_init_residual=True,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='NonLinearNeck',
        in_channels=2048,
        hid_channels=2048,
        out_channels=128,
        num_layers=2,
        with_avg_pool=True),
    head=dict(type='ContrastiveHead', temperature=0.1))
data_source = 'LDPolypVideo'
dataset_type = 'MultiViewDataset'
img_norm_cfg = dict(mean=[0.654, 0.424, 0.266], std=[0.257, 0.182, 0.148])
train_pipeline = [
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomHorizontalFlip'),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.8,
                contrast=0.8,
                saturation=0.8,
                hue=0.2)
        ],
        p=0.8),
    dict(type='RandomGrayscale', p=0.2),
    dict(type='GaussianBlur', sigma_min=0.1, sigma_max=2.0, p=0.5),
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
    train=dict(
        type='MultiViewDataset',
        data_source=dict(
            type='LDPolypVideo',
            data_prefix='/home/wsco/gty/datasets/LDPolypVideo/PreTrain',
            ann_file=None),
        num_views=[2],
        pipelines=[[{
            'type': 'RandomResizedCrop',
            'size': 224
        }, {
            'type': 'RandomHorizontalFlip'
        }, {
            'type':
            'RandomAppliedTrans',
            'transforms': [{
                'type': 'ColorJitter',
                'brightness': 0.8,
                'contrast': 0.8,
                'saturation': 0.8,
                'hue': 0.2
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
            'type': 'ToTensor'
        }, {
            'type': 'Normalize',
            'mean': [0.654, 0.424, 0.266],
            'std': [0.257, 0.182, 0.148]
        }]],
        prefetch=False))
optimizer = dict(
    type='LARS',
    lr=0.3,
    weight_decay=1e-06,
    momentum=0.9,
    paramwise_options=dict({
        '(bn|gn)(\d+)?.(weight|bias)':
        dict(weight_decay=0.0, lars_exclude=True),
        'bias':
        dict(weight_decay=0.0, lars_exclude=True)
    }))
optimizer_config = dict()
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0.0,
    warmup='linear',
    warmup_iters=10,
    warmup_ratio=0.0001,
    warmup_by_epoch=True)
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
work_dir = './work_dirs/selfsup/simclr_resnet50_4xb64-coslr-200e_ldpolypvideo'
auto_resume = False
gpu_ids = range(0, 4)
