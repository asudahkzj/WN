import json
import os
from sklearn.model_selection import train_test_split
from PIL import Image

with open('data/rvos/meta_expressions/train/meta_expressions.json', 'r') as f:
    content = json.load(f)
    content = content['videos']

# 对添加的数据随机分组
# files = list(content.keys())

# with open('data/rvos/instances_train_sub.json', 'r') as f:
#     ann = json.load(f)

# videos = []
# for i in ann['videos']:
#     name = i['file_names'][0].split('/')[0]
#     videos.append(name)

# # dif = list(set(files).difference(set(videos)))
# # print(dif)
# # print(len(dif))
# # print(len(videos))
# # print(len(files))

# train, test = train_test_split(files, test_size=0.25)
# test, valid = train_test_split(test, test_size=0.4)
# print(len(train), len(test), len(valid))

# data_plus = {'train': train, 'test': test, 'valid': valid}

# with open('data/rvos/data_plus_upset.json', 'w') as f:
#     json.dump(data_plus, f)

# 向注释文件中添加新数据
with open('data/rvos/data_plus.json', 'r') as f:
    dp = json.load(f)

train_p, test_p, valid_p = dp['train'], dp['test'], dp['valid']

with open('data/rvos/ann/instances_train_sub.json', 'r') as f:
    ann = json.load(f)

# print(train_p)

vid = len(ann['videos'])
videos = []
for file in train_p:
    vid += 1
    expressions = content[file]['expressions']
    frames = content[file]['frames']
    file_names = [file+'/'+frame+'.jpg' for frame in frames]
    img = Image.open(os.path.join('data/rvos/train/JPEGImages', file_names[0])) 
    video = {}
    video['width'] = img.width
    video['length'] = len(frames)
    video['data_captured'] = '2019-04-11 00:55:41.903902'
    video['license'] = 1
    video['flickr_url'] = ''
    video['file_names'] = file_names
    video['id'] = vid
    video['coco_url'] = ''
    video['height'] = img.height
    videos.append(video)

ann_plus = {}
ann_plus['info'] = ann['info']
ann_plus['licenses'] = ann['licenses']
ann_plus['videos'] = ann['videos'] + videos

with open('data/rvos/ann_plus/instances_train_sub_un.json', 'w') as f:
    json.dump(ann_plus, f)

print(len(ann_plus['videos']))

# 从头重新分数据
# with open('data/rvos/data_upset.json', 'r') as f:
#     dp = json.load(f)
    
# train_p, test_p, valid_p = dp['train'], dp['test'], dp['valid']

# with open('data/rvos/ann/instances_test_sub.json', 'r') as f:
#     ann = json.load(f)

# vid = 0
# videos = []
# for file in test_p:
#     vid += 1
#     expressions = content[file]['expressions']
#     frames = content[file]['frames']
#     file_names = [file+'/'+frame+'.jpg' for frame in frames]
#     img = Image.open(os.path.join('data/rvos/train/JPEGImages', file_names[0])) 
#     video = {}
#     video['width'] = img.width
#     video['length'] = len(frames)
#     video['data_captured'] = '2019-04-11 00:55:41.903902'
#     video['license'] = 1
#     video['flickr_url'] = ''
#     video['file_names'] = file_names
#     video['id'] = vid
#     video['coco_url'] = ''
#     video['height'] = img.height
#     videos.append(video)

# ann_upset = {}
# ann_upset['info'] = ann['info']
# ann_upset['licenses'] = ann['licenses']
# ann_upset['videos'] = videos

# with open('data/rvos/ann_upset/instances_test_sub_un.json', 'w') as f:
#     json.dump(ann_upset, f)