from FastMTCNN import FastMTCNN
from PIL import Image
import torch
from imutils.video import FileVideoStream
import cv2
import time
import glob
from tqdm import tqdm
from facenet_pytorch import MTCNN
import numpy as np
import os


start = 60000
stop = 90000

device = 'cuda' if torch.cuda.is_available() else 'cpu'

filenames = glob.glob('/home/aelbakry1999/VoxCeleb2/dev/mp4/**/*.mp4', recursive = True)[start:stop]


fast_mtcnn = FastMTCNN(
    stride=1,
    resize=1,
    margin=40,
    factor=0.6,
    keep_all=True,
    device=device
)


mtcnn = MTCNN()


def run_detection(fast_mtcnn, filenames):
    frames = []
    frames_processed = 0
    faces_detected = 0
    batch_size = 64
    start = time.time()
    video_index = start

    for filename in tqdm(filenames):

        v_cap = FileVideoStream(filename).start()
        v_lent = int(v_cap.stream.get(cv2.CAP_PROP_FRAME_COUNT))
        v_len = 15

        if v_lent < v_len:
            v_len = v_lent

        for j in range(v_len):

            frame = v_cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

            if len(frames) >= batch_size or j == v_len - 1:

                fname = os.sep.join(filename.split(os.sep)[-2:])
                fname = os.path.splitext(fname)[0]

                save_paths = [f'images/VoxCeleb2/video_{video_index}/frame{i}.jpeg' for i in range(len(frames))]

                faces = fast_mtcnn(frames)

                if len(faces) > 0:
                    mtcnn(faces, save_paths)

                frames_processed += len(frames)
                faces_detected += len(faces)
                frames = []



                # print(
                #     f'Frames per second: {frames_processed / (time.time() - start):.3f},',
                #     f'faces detected: {faces_detected}\r',
                #     end=''
                # )
        video_index += 1
        v_cap.stop()

run_detection(fast_mtcnn, filenames)
