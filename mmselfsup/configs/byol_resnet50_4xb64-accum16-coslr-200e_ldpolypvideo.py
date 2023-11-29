model = dict(
    type='BYOL',
    base_momentum=0.99,
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=[4],
        norm_cfg=dict(type='SyncBN'),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='NonLinearNeck',
        in_channels=2048,
        hid_channels=4096,
        out_channels=256,
        num_layers=2,
        with_bias=True,
        with_last_bn=False,
        with_avg_pool=True),
    head=dict(
        type='LatentPredictHead',
        predictor=dict(
            type='NonLinearNeck',
            in_channels=256,
            hid_channels=4096,
            out_channels=256,
            num_layers=2,
            with_bias=True,
            with_last_bn=False,
            with_avg_pool=False)))
data_source = 'LDPolypVideo'
dataset_type = 'MultiViewDataset'
img_norm_cfg = dict(mean=[0.654, 0.424, 0.266], std=[0.257, 0.182, 0.148])
train_pipeline1 = [
    dict(type='RandomResizedCrop', size=224, interpolation=3),
    dict(type='RandomHorizontalFlip'),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1)
        ],
        p=0.8),
    dict(type='RandomGrayscale', p=0.2),
    dict(type='GaussianBlur', sigma_min=0.1, sigma_max=2.0, p=1.0),
    dict(type='Solarization', p=0.0),
    dict(type='ToTensor'),
    dict(
        type='Normalize',
        mean=[0.654, 0.424, 0.266],
        std=[0.257, 0.182, 0.148])
]
train_pipeline2 = [
    dict(type='RandomResizedCrop', size=224, interpolation=3),
    dict(type='RandomHorizontalFlip'),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1)
        ],
        p=0.8),
    dict(type='RandomGrayscale', p=0.2),
    dict(type='GaussianBlur', sigma_min=0.1, sigma_max=2.0, p=0.1),
    dict(type='Solarization', p=0.2),
    dict(type='ToTensor'),
    dict(
        type='Normalize',
        mean=[0.654, 0.424, 0.266],
        std=[0.257, 0.182, 0.148])
]
prefetch = False
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=8,
    train=dict(
        type='MultiViewDataset',
        data_source=dict(
            type='LDPolypVideo',
            data_prefix='/home/wsco/gty/datasets/LDPolypVideo/PreTrain',
            ann_file=None),
        num_views=[1, 1],
        pipelines=[[{
            'type': 'RandomResizedCrop',
            'size': 224,
            'interpolation': 3
        }, {
            'type': 'RandomHorizontalFlip'
        }, {
            'type':
            'RandomAppliedTrans',
            'transforms': [{
                'type': 'ColorJitter',
                'brightness': 0.4,
                'contrast': 0.4,
                'saturation': 0.2,
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
            'p': 1.0
        }, {
            'type': 'Solarization',
            'p': 0.0
        }, {
            'type': 'ToTensor'
        }, {
            'type': 'Normalize',
            'mean': [0.654, 0.424, 0.266],
            'std': [0.257, 0.182, 0.148]
        }],
                   [{
                       'type': 'RandomResizedCrop',
                       'size': 224,
                       'interpolation': 3
                   }, {
                       'type': 'RandomHorizontalFlip'
                   }, {
                       'type':
                       'RandomAppliedTrans',
                       'transforms': [{
                           'type': 'ColorJitter',
                           'brightness': 0.4,
                           'contrast': 0.4,
                           'saturation': 0.2,
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
                       'p': 0.1
                   }, {
                       'type': 'Solarization',
                       'p': 0.2
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
    lr=4.8,
    weight_decay=1e-06,
    momentum=0.9,
    paramwise_options=dict({
        '(bn|gn)(\d+)?.(weight|bias)':
        dict(weight_decay=0.0, lars_exclude=True),
        'bias':
        dict(weight_decay=0.0, lars_exclude=True)
    }))
optimizer_config = dict(update_interval=16)
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
update_interval = 16
custom_hooks = [dict(type='BYOLHook', end_momentum=1.0, update_interval=16)]
work_dir = './work_dirs/selfsup/byol_resnet50_4xb64-accum16-coslr-200e_ldpolypvideo'
auto_resume = False
gpu_ids = range(0, 4)
