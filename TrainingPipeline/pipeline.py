import pandas as pd
import numpy as np
import cv2
import os
from tqdm import tqdm,trange
from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face
import torch
from torchvision.transforms import ToTensor


# import torch.nn as nn
# import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")

max_frames = 10
df_train = pd.read_json('/home/aelbakry1999/sample_data/train_sample_videos/metadata.json')
LABELS = ['REAL','FAKE']
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()




def get_paths(x):
    image_paths=[]

    for num in range(max_frames):
        path = '/home/aelbakry1999/images/train_sample_videos/'+ x.replace('.mp4', '') + '/frame' + str(num) +'.jpeg'
        image_paths.append(path)
        if not os.path.exists(path):
            raise Exception
    return image_paths

def read_img(path):
    frames = []
    for i in range(max_frames):
        frames.append(cv2.cvtColor(cv2.imread(path[i]),cv2.COLOR_BGR2RGB))
    return frames

# def process_faces(faces, resnet):
#     # Filter out frames without faces
#     faces = torch.cat(torch.from_numpy(faces)).to(device)
#
#     # Generate facial feature vectors using a pretrained model
#     embeddings = resnet(faces)
#
#     # Calculate centroid for video and distance of each face's feature vector from centroid
#     # centroid = embeddings.mean(dim=0)
#     # x = (embeddings - centroid).norm(dim=1).cpu().numpy()
#
#     return embeddings.numpy()

paths=[]
y=[]
images = list(df_train.columns.values)
for x in images:
    try:
        paths.append(get_paths(x))
        y.append(LABELS.index(df_train[x]['label']))
    except Exception as err:
        # print(err)
        pass

print(np.shape(paths))
print(np.shape(y))

X=[]
for img in tqdm(paths):
    X.append(read_img(img))

print(np.shape(X))

X_embedded=[]


tf_img = lambda i: ToTensor()(i).unsqueeze(0)
embeddings = lambda input: resnet(input)

list_embs = []
with torch.no_grad():
    for faces in tqdm(X):
        t = tf_img(faces[0]).to(device)
        e = embeddings(t).squeeze().cpu().tolist()
        list_embs.append(e)


print(np.shape(list_embs))
