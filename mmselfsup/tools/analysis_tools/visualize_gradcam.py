from torchvision.models import resnet50
from torchcam.methods import GradCAM
import torch
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
import numpy as np
from PIL import Image


def overlay_mask(img: Image.Image, mask: Image.Image):
    if not isinstance(img, Image.Image) or not isinstance(mask, Image.Image):
        raise TypeError("img and mask arguments need to be PIL.Image")

    overlay = np.asarray(mask.resize(img.size, resample=Image.BICUBIC))
    overlay = np.maximum(overlay, 0)
    overlay = overlay / overlay.max()

    overlay = Image.fromarray(255 * overlay ** 2).convert('RGB')

    overlayed_img = Image.fromarray(np.multiply(np.asarray(img), np.asarray(overlay) / 255.).astype(np.uint8))

    return overlayed_img


if __name__ == '__main__':
    model = resnet50(pretrained=False).eval()
    state_dict = torch.load('./backbone_weights/detector_backbone/faster_rcnn_r50_fpn_1x_ldpolypvideo_best.pth')['state_dict']
    model.load_state_dict(state_dict, strict=False)
    cam_extractor = GradCAM(model)

    for i in range(1, 13):
        # Get your input
        img = read_image(f"./demo/visualize_ldpolypvideo/{i}.jpg")
        # Preprocess it for your chosen model
        input_tensor = normalize(resize(img, (224, 224)) / 255., [0.654, 0.424, 0.266], [0.257, 0.182, 0.148])

        # Preprocess your data and feed it to the model
        out = model(input_tensor.unsqueeze(0))
        # Retrieve the CAM by passing the class index and the model output
        activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)

        # Resize the CAM and overlay it
        result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(0), mode='F'))
        # save it
        result.save(f'./demo/visualize_ldpolypvideo/{i}-imgnt.jpg')
