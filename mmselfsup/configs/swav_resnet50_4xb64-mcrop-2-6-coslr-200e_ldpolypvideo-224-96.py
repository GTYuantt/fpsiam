model = dict(
    type='SwAV',
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=[4],
        norm_cfg=dict(type='SyncBN'),
        zero_init_residual=True,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='SwAVNeck',
        in_channels=2048,
        hid_channels=2048,
        out_channels=128,
        with_avg_pool=True),
    head=dict(
        type='SwAVHead',
        feat_dim=128,
        epsilon=0.05,
        temperature=0.1,
        num_crops=[2, 6]))
data_source = 'LDPolypVideo'
dataset_type = 'MultiViewDataset'
img_norm_cfg = dict(mean=[0.654, 0.424, 0.266], std=[0.257, 0.182, 0.148])
num_crops = [2, 6]
color_distort_strength = 1.0
train_pipeline1 = [
    dict(type='RandomResizedCrop', size=224, scale=(0.14, 1.0)),
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
    dict(type='RandomHorizontalFlip', p=0.5),
    dict(type='ToTensor'),
    dict(
        type='Normalize',
        mean=[0.654, 0.424, 0.266],
        std=[0.257, 0.182, 0.148])
]
train_pipeline2 = [
    dict(type='RandomResizedCrop', size=96, scale=(0.05, 0.14)),
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
    dict(type='RandomHorizontalFlip', p=0.5),
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
        num_views=[2, 6],
        pipelines=[[{
            'type': 'RandomResizedCrop',
            'size': 224,
            'scale': (0.14, 1.0)
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
            'type': 'RandomHorizontalFlip',
            'p': 0.5
        }, {
            'type': 'ToTensor'
        }, {
            'type': 'Normalize',
            'mean': [0.654, 0.424, 0.266],
            'std': [0.257, 0.182, 0.148]
        }],
                   [{
                       'type': 'RandomResizedCrop',
                       'size': 96,
                       'scale': (0.05, 0.14)
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
                       'type': 'RandomHorizontalFlip',
                       'p': 0.5
                   }, {
                       'type': 'ToTensor'
                   }, {
                       'type': 'Normalize',
                       'mean': [0.654, 0.424, 0.266],
                       'std': [0.257, 0.182, 0.148]
                   }]],
        prefetch=False))
optimizer = dict(type='LARS', lr=0.6, weight_decay=1e-06, momentum=0.9)
optimizer_config = dict(frozen_layers_cfg=dict(prototypes=5005))
lr_config = dict(policy='CosineAnnealing', min_lr=0.0006)
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
custom_hooks = [
    dict(
        type='SwAVHook',
        priority='VERY_HIGH',
        batch_size=64,
        epoch_queue_starts=15,
        crops_for_assign=[0, 1],
        feat_dim=128,
        queue_length=3840)
]
work_dir = './work_dirs/selfsup/swav_resnet50_4xb64-mcrop-2-6-coslr-200e_ldpolypvideo-224-96'
auto_resume = False
gpu_ids = range(0, 4)
