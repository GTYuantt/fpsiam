import os
import os.path as osp
import mmcv


def convert_ldpolypvideo_to_coco(root_dir, out_file):
    ann_dir = osp.join(root_dir, 'Annotations')
    img_dir = osp.join(root_dir, 'Images')

    images = []
    annotations = []
    categories = [
        {
            "supercategory": "none",
            "id": 0,
            "name": "polyp"
        }
    ]

    img_id = 0
    ann_id = 0
    for video in os.listdir(img_dir):
        for frame in os.listdir(osp.join(img_dir, video)):
            image = {
                "id": img_id,
                "width": 560,
                "height": 480,
                "file_name": osp.join(img_dir, video, frame)
            }
            images.append(image)

            ann_file = osp.join(ann_dir, video, frame.split('.')[0].split('_')[2] + '.txt')
            with open(ann_file, 'r') as f:
                lines = f.readlines()
                polyp_num = lines[0].strip()
                if polyp_num != '0':
                    polyps = lines[1:]
                    for polyp in polyps:
                        pos = polyp.strip().split(' ')
                        xmin = int(pos[0])
                        ymin = int(pos[1])
                        xmax = int(pos[2])
                        ymax = int(pos[3])
                        annotation = {
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": 0,
                            "area": float((xmax - xmin) * (ymax - ymin)),
                            "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                            "iscrowd": 0
                        }
                        annotations.append(annotation)
                        ann_id += 1

            img_id += 1

    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=categories)

    mmcv.dump(coco_format_json, out_file)


if __name__ == '__main__':
    convert_ldpolypvideo_to_coco('/home/wsco/gty/datasets/LDPolypVideo/TrainValid',
                                 '/home/wsco/gty/datasets/LDPolypVideo/TrainValid/annotations_coco.json')
    convert_ldpolypvideo_to_coco('/home/wsco/gty/datasets/LDPolypVideo/Test',
                                 '/home/wsco/gty/datasets/LDPolypVideo/Test/annotations_coco.json')

