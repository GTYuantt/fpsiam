import torch.nn as nn

from ..builder import ALGORITHMS, build_backbone, build_head, build_neck
from .base import BaseModel


@ALGORITHMS.register_module()
class FPSiam(BaseModel):
    """FPSiam.

    Implementation of `Self-supervised representation learning using feature pyramid siamese networks for colorectal polyp detection`.
    The operation of fixing learning rate of predictor is the same as SimSiam, implemented in `core/hooks/simsiam_hook.py`.

    Args:
        lamda (float): weight to balance the global and local loss.
        backbone (dict): Config dict for module of backbone.
        neck (dict): Config dict for module of deep features to compact
            feature vectors. Defaults to None.
        head (dict): Config dict for module of loss functions.
            Defaults to None.
    """

    def __init__(self,
                 lamda,
                 backbone,
                 neck=None,
                 head=None,
                 init_cfg=None,
                 **kwargs):
        super(FPSiam, self).__init__(init_cfg)
        assert neck is not None
        self.lamda = lamda
        self.encoder = nn.Sequential(
            build_backbone(backbone), build_neck(neck))
        self.backbone = self.encoder[0]
        self.neck = self.encoder[1]
        assert head is not None
        self.head = build_head(head)

    def extract_feat(self, img):
        """Function to extract features from backbone.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            tuple[Tensor]: backbone outputs.
        """
        x = self.backbone(img)
        return x

    def forward_train(self, img):
        """Forward computation during training.

        Args:
            img (list[Tensor]): A list of input images with shape
                (N, C, H, W). Typically these should be mean centered
                and std scaled.
        Returns:
            loss[str, Tensor]: A dictionary of loss components
        """

        assert isinstance(img, list)

        img_v1 = img[0]
        img_v2 = img[1]

        z1_g, z1_l = self.encoder(img_v1)
        z2_g, z2_l = self.encoder(img_v2)

        losses = dict()
        losses['loss_local'] = 0.5 * (self.head(z1_l, z2_l)['loss'] + self.head(z2_l, z1_l)['loss'])
        losses['loss_global'] = 0.5 * (self.head(z1_g, z2_g)['loss'] + self.head(z2_g, z1_g)['loss'])
        losses['loss'] = self.lamda * losses['loss_local'] + (1 - self.lamda) * losses['loss_global']

        return losses
