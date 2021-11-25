import os
import numpy as np
import torch
from PIL import Image

mask_path = 'data/rvos/train/Annotations/b0623a6232/00000.png'
mask = Image.open(mask_path)
# mask = np.array(mask)
mask = torch.tensor(mask)
# obj_ids = np.unique(mask)
print(mask)
mask = mask == 3
print(mask)
# print(mask.shape)
# print(obj_ids)
pos = np.where(mask)
# xmin = np.min(pos[1])
# xmax = np.max(pos[1])
# ymin = np.min(pos[0])
# ymax = np.max(pos[0])
print(pos)
# print(xmin, xmax, ymin, ymax)
# print(np.sum(mask))
print(torch.ones((5,), dtype=torch.int64))