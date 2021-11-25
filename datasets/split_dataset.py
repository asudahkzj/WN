import json
from os import WIFSTOPPED

# with open('data/rvos/meta_expressions/train/meta_expressions.json', 'r') as f:
#     content = json.load(f)
#     content = content['videos']

# files = list(content.keys())

# count = 0
# for a in content.values():
#     b = a['expressions']
#     c = list(b.values())
#     count += int(c[-1]['obj_id'])

# print(count)  # 6489




with open('data/rvos/instances_train_sub.json', 'r') as f:
    ann = json.load(f)
    
videos = []
for i in ann['videos']:
    name = i['file_names'][0].split('/')[0]
    videos.append(name)

print(len(videos))
# ins = list(set(files).intersection(videos))
# print(len(ins))

from sklearn.model_selection import train_test_split

# train:test:valid = 3471:305:202

train, test = train_test_split(videos, test_size=0.25)

test, valid = train_test_split(test, test_size=0.4)

print(len(train), len(test), len(valid))

train, test, valid = set(train), set(test), set(valid)  # 1678：336：224

v_count_train, v_count_test, v_count_valid = 1, 1, 1
a_count_train, a_count_test, a_count_valid = 1, 1, 1
a_id = 0

ann_train, ann_test, ann_valid = {}, {}, {}
ann_test['info'] = ann['info']
ann_train['info'] = ann['info']
ann_valid['info'] = ann['info']
ann_test['licenses'] = ann['licenses']
ann_train['licenses'] = ann['licenses']
ann_valid['licenses'] = ann['licenses']
ann_test_videos, ann_valid_videos, ann_train_videos = [], [], []
ann_test_anns, ann_valid_anns, ann_train_anns = [], [], [] 

annotations = ann['annotations']
a_l = len(annotations)
for video in ann['videos']:
    name = video['file_names'][0].split('/')[0]
    id = video['id']
    if name in test:
        v = video
        v['id'] = v_count_test        
        ann_test_videos.append(v)
        while a_id < a_l:
            annotation = annotations[a_id]
            if annotation['video_id'] == id:
                a = annotation
                a['video_id'] = v_count_test
                a['id'] = a_count_test
                a_count_test += 1
                ann_test_anns.append(a)
                a_id += 1
            else:
                break
        v_count_test += 1  
    elif name in valid:
        v = video
        v['id'] = v_count_valid       
        ann_valid_videos.append(v)
        while a_id < a_l:
            annotation = annotations[a_id]
            if annotation['video_id'] == id:
                a = annotation
                a['video_id'] = v_count_valid
                a['id'] = a_count_valid
                a_count_valid += 1
                ann_valid_anns.append(a)
                a_id += 1
            else:
                break
        v_count_valid += 1
    else:
        v = video
        v['id'] = v_count_train      
        ann_train_videos.append(v)
        while a_id < a_l:
            annotation = annotations[a_id]
            if annotation['video_id'] == id:
                a = annotation
                a['video_id'] = v_count_train
                a['id'] = a_count_train
                a_count_train += 1
                ann_train_anns.append(a)
                a_id += 1
            else:
                break
        v_count_train += 1


ann_test['videos'] = ann_test_videos
ann_test['annotations'] = ann_test_anns
ann_test['categories'] = ann['categories']

ann_train['videos'] = ann_train_videos
ann_train['annotations'] = ann_train_anns
ann_train['categories'] = ann['categories']

ann_valid['videos'] = ann_valid_videos
ann_valid['annotations'] = ann_valid_anns
ann_valid['categories'] = ann['categories']

with open('data/rvos/ann/instances_train_sub.json', 'w') as f:
    json.dump(ann_train, f)

with open('data/rvos/ann/instances_test_sub.json', 'w') as f:
    json.dump(ann_test, f) 

with open('data/rvos/ann/instances_valid_sub.json', 'w') as f:
    json.dump(ann_valid, f)