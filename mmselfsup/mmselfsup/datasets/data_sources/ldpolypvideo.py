import os

import mmcv
import os.path as osp
import random

import numpy as np
from PIL import Image

from ..builder import DATASOURCES
from .base import BaseDataSource


@DATASOURCES.register_module()
class LDPolypVideo(BaseDataSource):

    def __init__(self,
                 data_prefix,
                 classes=None,
                 ann_file=None,
                 test_mode=False,
                 color_type='color',
                 channel_order='rgb',
                 file_client_args=dict(backend='disk')):
        self.idx_dict = {}
        super().__init__(data_prefix, classes, ann_file, test_mode, color_type, channel_order, file_client_args)

    def load_annotations(self):

        assert isinstance(self.data_prefix, str)
        data_vp_prefix = f'{self.data_prefix}/VideosWithPolyps/Images'
        data_vn_prefix = f'{self.data_prefix}/VideosWithoutPolyps/Images'

        data_infos = []
        idx_dict = {}
        idx = 0
        for data_vp in os.listdir(data_vp_prefix):
            for data in os.listdir(f"{data_vp_prefix}/{data_vp}"):
                info = {'img_prefix': f"{data_vp_prefix}/{data_vp}",
                        'img_info': {'filename': data},
                        'video_type': 'vp',
                        'video_idx': data_vp,
                        'img_idx': data.split('.')[0].split('_')[2]}
                data_infos.append(info)
                idx_dict[data.split('.')[0]] = idx
                idx += 1

        for data_vn in os.listdir(data_vn_prefix):
            for data in os.listdir(f"{data_vn_prefix}/{data_vn}"):
                info = {'img_prefix': f"{data_vn_prefix}/{data_vn}",
                        'img_info': {'filename': data},
                        'video_type': 'vn',
                        'video_idx': data_vn,
                        'img_idx': data.split('.')[0].split('_')[2]}
                data_infos.append(info)
                idx_dict[data.split('.')[0]] = idx
                idx += 1

        self.idx_dict = idx_dict

        return data_infos

    def get_neighbor_img_pair(self, idx, neighbor_range):
        video_type = self.data_infos[idx]['video_type']
        video_idx = self.data_infos[idx]['video_idx']
        img_idx = self.data_infos[idx]['img_idx']

        while True:
            bias = random.randint(-neighbor_range, neighbor_range)
            neighbor_img_idx = int(img_idx) + bias
            key = f"{video_type}_{video_idx}_{neighbor_img_idx}"
            if key in self.idx_dict.keys():
                neighbor_idx = self.idx_dict[key]
                break

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        filename = osp.join(
            self.data_infos[idx]['img_prefix'],
            self.data_infos[idx]['img_info']['filename'])
        neighbor_filename = osp.join(
            self.data_infos[neighbor_idx]['img_prefix'],
            self.data_infos[neighbor_idx]['img_info']['filename'])

        img_bytes = self.file_client.get(filename)
        neighbor_img_bytes = self.file_client.get(neighbor_filename)

        img = mmcv.imfrombytes(
            img_bytes,
            flag=self.color_type,
            channel_order=self.channel_order)
        neighbor_img = mmcv.imfrombytes(
            neighbor_img_bytes,
            flag=self.color_type,
            channel_order=self.channel_order)

        img = img.astype(np.uint8)
        neighbor_img = neighbor_img.astype(np.uint8)

        return [Image.fromarray(img), Image.fromarray(neighbor_img)]
