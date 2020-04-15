# import pandas as pd
import numpy as np
import cv2
import os
import glob

from tqdm import tqdm,trange
# from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face
# import torch
# from torchvision.transforms import ToTensor
# from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D, Bidirectional
# from keras.layers.recurrent import LSTM
# from keras.models import Sequential
# from keras.optimizers import Adam, RMSprop, SGD
# from keras.layers.wrappers import TimeDistributed
# from keras.layers.convolutional import (Conv2D, MaxPooling3D, Conv3D,
#     MaxPooling2D)
# from keras.utils import to_categorical
# import matplotlib.pyplot as plt
# import matplotlib

youtube_faces = sorted(os.listdir('/home/aelbakry1999/YouTubeFaces/aligned_images_DB'))
youtube_faces_path = '/home/aelbakry1999/YouTubeFaces/aligned_images_DB'
X_ = []
y_ = []
for dirpaths in tqdm(youtube_faces):
    dirpath = sorted(os.listdir(os.path.join(youtube_faces_path, dirpaths)))
    for subdirpaths in dirpath:
        subdirpath = sorted(os.listdir(os.path.join(youtube_faces_path, dirpaths, subdirpaths)))[:10]
        frames_path = []
        for filename in subdirpath:
            path = os.path.join(youtube_faces_path, dirpaths, subdirpaths, filename)
            frames_path.append(cv2.resize(cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB), (160, 160)))
        y_.append(0)
        X_.append(frames_path)



print(np.shape(X_))
print(np.shape(y_))
