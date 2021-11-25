#coding=utf-8
#测试文件
from numpy.core.defchararray import count
from sigprocess import *
from calcmfcc import *
import scipy.io.wavfile as wav
import numpy as np
import os
import traceback
import warnings

# warnings.filterwarnings('error') 

(rate,sig) = wav.read("data/audio_raw/003234408d_8.wav")
mfcc_feat = calcMFCC_delta_delta(sig,rate) 
print(mfcc_feat.shape)

raw_path = "data/audio_raw"
filenames = os.listdir(raw_path)
print(len(filenames))

# raw_path = "data/audio_feature"
# filenames = os.listdir(raw_path)
# print(len(filenames))

count = 0

for filename in filenames:
    count += 1
    if count % 100 == 0:
        print(count)
    try:
        name = filename.split('.')[0]
        (rate,sig) = wav.read(os.path.join(raw_path, filename))
        mfcc_feat = calcMFCC_delta_delta(sig,rate) 
        np.save("data/audio_feature/"+name+'.npy', mfcc_feat)
    except Exception as e:
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^", filename)
        traceback.print_exc()
