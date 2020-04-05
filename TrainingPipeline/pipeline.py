import pandas as pd
import numpy as np
import cv2
import os
from tqdm import tqdm,trange

# import torch.nn as nn
# import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")

max_frames = 10
df_train = pd.read_json('/home/aelbakry1999/sample_data/train_sample_videos/metadata.json')
LABELS = ['REAL','FAKE']



def get_paths(x):
    image_paths=[]

    for num in range(max_frames):
        path = '/home/aelbakry1999/images/train_sample_videos/'+ x.replace('.mp4', '') + '/frame' + str(num) +'.jpeg'
        image_paths.append(path)
        if not os.path.exists(path):
            raise Exception
    return image_paths

def read_img(path):
    return cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)


paths=[]
y=[]
images = list(df_train.columns.values)
for x in images:
    try:
        paths.append(get_paths(x))
        y.append(LABELS.index(df_train[x]['label']))
    except Exception as err:
        print(err)
        pass

print(np.shape(paths))
print(np.shape(y))

X=[]
for img in tqdm(paths):
    X.append(read_img(img[0]))
