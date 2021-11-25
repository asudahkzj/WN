import json
from PIL import Image
import os
import numpy as np
import torch
from skimage import measure
from pycocotools import mask as coco_mask
from itertools import groupby


def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation
    Args:
    binary_mask: a 2D binary numpy array where '1's represent the object
    tolerance: Maximum distance from original points of polygon to approximated
    polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """
    

    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        if not polygons:
            mask = torch.zeros((height,width), dtype=torch.uint8)
        else:
            rles = coco_mask.frPyObjects(polygons, height, width)
            mask = coco_mask.decode(rles)
            if len(mask.shape) < 3:
                mask = mask[..., None]
            mask = torch.as_tensor(mask, dtype=torch.uint8)
            mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
                counts.append(0)
        counts.append(len(list(elements)))

    return rle


with open('data/rvos/ann_plus/instances_train_sub_un.json', 'r') as f:
    ann = json.load(f)

with open('data/rvos/meta_expressions/train/meta_expressions.json', 'r') as f:
    content = json.load(f)
    content = content['videos']

ann_path = 'data/rvos/train/Annotations'

count_id = 1
annotations = []

for i in ann['videos']:    
    video_id = i['id']
    name = i['file_names'][0].split('/')[0]
    expressions = content[name]['expressions']
    obj_ids = set()
    for v in expressions.values():
        obj_ids.add(int(v['obj_id']))
    obj_ids = list(obj_ids)
    obj_ids.sort()
    for j in range(obj_ids[-1]):
        ids = j+1
        annotation = {}
        if ids in obj_ids:            
            annotation['height'] = i['height']
            annotation['width'] = i['width']
            annotation['length'] = 1
            annotation['category_id'] = 1
            segs, boxes, areas = [], [], []
            for frame in i['file_names']:
                mask_path = os.path.join(str(ann_path), frame[:-3]+'png')
                mask = np.array(Image.open(mask_path))    
                mask = (mask == ids).astype(int)
                pos = np.where(mask)
                if len(pos[0]) == 0:
                    boxes.append(None)
                    segs.append(None)
                    areas.append(None)
                else:
                    xmin = np.min(pos[1]).tolist()
                    xmax = np.max(pos[1]).tolist()
                    ymin = np.min(pos[0]).tolist()
                    ymax = np.max(pos[0]).tolist()
                    boxes.append([xmin, ymin, xmax, ymax])
                    seg = binary_mask_to_rle(mask)
                    segs.append(seg)
                    areas.append(np.sum(mask).tolist())
                # mask2 = convert_coco_poly_to_mask([seg], 720, 1280)[0].numpy()
                # print((mask1==mask2).all())
            annotation['segmentations'] = segs
            annotation['bboxes'] = boxes
            annotation['video_id'] = video_id
            if video_id % 10 == 0:
                print(video_id)
            annotation['iscrowd'] = 0
            annotation['id'] = count_id
            count_id += 1
            annotation['areas'] = areas
        else:
            annotation['height'] = i['height']
            annotation['width'] = i['width']
            annotation['length'] = 1
            annotation['category_id'] = 0
            segs, boxes, areas = [], [], []
            annotation['segmentations'] = segs
            annotation['bboxes'] = boxes
            annotation['video_id'] = video_id
            annotation['iscrowd'] = 0
            annotation['id'] = count_id
            count_id += 1
            annotation['areas'] = areas
        annotations.append(annotation)
    # break

new_ann = {}
new_ann['info'] = ann['info']
new_ann['licenses'] = ann['licenses']
new_ann['videos'] = ann['videos']
new_ann['annotations'] = annotations
# new_ann['categories'] = ann['categories']

with open('data/rvos/ann_plus/instances_train_sub.json', 'w') as f:
    json.dump(new_ann, f)

print("finish!")