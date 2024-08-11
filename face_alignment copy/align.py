import sys
import os

from face_alignment import mtcnn
from face_alignment.mtcnn_pytorch.src import show_bboxes
import argparse
from PIL import Image
from tqdm import tqdm
import random
from datetime import datetime
from PIL import ImageDraw
from PIL import Image
import IPython.display as display
import matplotlib.pyplot as plt
import torch
import time

from insightface.utils import face_align
from insightface.app import FaceAnalysis
import cupy as cp
import numpy as np
import cv2


mtcnn_model = mtcnn.MTCNN(device='cuda:0', crop_size=(112, 112))
detector = FaceAnalysis(name ='buffalo_s',providers =['CUDAExecutionProvider','CPUExecutionProvider'])
detector.prepare(ctx_id= 0,det_size=(640,640))

def show_bboxes(img, bounding_boxes, facial_landmarks=[]):
    """Draw bounding boxes and facial landmarks.

    Arguments:
        img: an instance of PIL.Image.
        bounding_boxes: a float numpy array of shape [n, 5].
        facial_landmarks: a float numpy array of shape [n, 10].

    Returns:
        an instance of PIL.Image.
    """

    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)

    for b in bounding_boxes:
        draw.rectangle([
            (b[0], b[1]), (b[2], b[3])
        ], outline='white')

    for p in facial_landmarks:
        for i in range(5):
            draw.ellipse([
                (p[i] - 1.0, p[i + 5] - 1.0),
                (p[i] + 1.0, p[i + 5] + 1.0)
            ], outline='blue')

    return img_copy


def add_padding(pil_img, top, right, bottom, left, color=(0,0,0)):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

def get_aligned_face(image_path, source, rgb_pil_image=None):
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    time1 = time.time()
    count = 0
    if rgb_pil_image is None:
        img = Image.open(image_path).convert('RGB')
    else:
        assert isinstance(rgb_pil_image, Image.Image), 'Face alignment module requires PIL image or path to the image'
        img = rgb_pil_image
    # find face
    time2 = time.time()
    try:
        # bboxes, faces = mtcnn_model.align_multi(img=img, limit=2)
        bboxes, faces = [], []
        face_detector = detector.get(np.asarray(img))
        for each_face in face_detector:
            aligned_face = norm_crop(img=img, landmark=faces['kps'], image_size=112, mode='arcface')
            bboxes.append(faces['bbox'])
            faces.append(aligned_face)

    except Exception as e:
        print('Face detection Failed due to error.')
        print(e)
        face = None
    time3 = time.time()
    if source == 'database':
        time4 = time.time()
        print("HERE:", time4-time3, time3-time2, time2-time1)
        return faces[0], bboxes

    elif source == 'frame':
        time4 = time.time()
        print("HERE:", time4-time3, time3-time2, time2-time1)
        return faces, bboxes

def norm_crop(img, landmark, image_size=112, mode='arcface'):
    M = torch.tensor(face_align.estimate_norm(landmark, image_size, mode)).float().cuda()
    warped = cv2.warpAffine(np.asarray(img),M.cpu().detach().numpy(), (image_size, image_size), borderValue=0.0)
    return warped

    
