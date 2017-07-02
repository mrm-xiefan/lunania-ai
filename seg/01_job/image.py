#!/usr/bin/env python35
# -*- coding: utf-8 -*-

import argparse
import os
import shutil
from pathlib import Path
import logging.config
from luna import LunaExcepion

logging.config.fileConfig("logging.conf")
logger = logging.getLogger()

import numpy as np
from PIL import Image

# keras CPU only  mode.
#os.environ["CUDA_VISIBLE_DEVICES"] = ""


def show_anotation_data(file_path, dist_dir, dist_file_name: str):

    dist_file_path = os.path.join(dist_dir, dist_file_name)

    png = np.asarray(Image.open(file_path))
    shape = png.shape
    print(shape)
    dist_img = np.zeros([shape[0],shape[1],3])
    
    height = shape[0]
    width = shape[1]
    for h in range(height):
        for w in range(width):
            if png[h,w][0] == 0:
                dist_img[h,w] = [255,255,255]
            elif png[h,w][0] == 1:
                dist_img[h,w] = [255,255,100]
            elif png[h,w][0] == 2:
                dist_img[h,w] = [100,255,255]
            else:
                dist_img[h,w] = [255,100,255]

    im = Image.fromarray(np.uint8(dist_img))
    im.save(dist_file_path)

def resize_one_jpg(full_name):

    img_array = np.asarray(Image.open(full_name))
    file_name, ext = os.path.splitext(os.path.basename(full_name))
    dist_img = np.zeros([500, 500, 3])
    dist_img[0:img_array.shape[0], 0:img_array.shape[1], 0:img_array.shape[2]] = img_array
    new_file = "/opt/fr/other/tools/VOC2012/resizedjpg/" + file_name + ".jpg"
    im = Image.fromarray(np.uint8(dist_img))
    im.save(new_file)

def resize_jpg():

    file_path = Path("/opt/fr/other/tools/VOC2012/SegmentationClass")
    for i, image_file in enumerate(file_path.glob('**/*.png')):
        full_name = str(image_file)
        file_name, ext = os.path.splitext(os.path.basename(full_name))
        resize_one_jpg("/opt/fr/other/tools/VOC2012/JPEGImages/" + file_name + ".jpg")

def save_png_to_txt(file_path):

    #file_path = Path(file_path)
    #max_height = 0
    #max_width = 0
    #for i, path in enumerate(file_path.glob('**/*.jpg')):
    #    #img_array = np.load(path)
    #    img_array = np.asarray(Image.open(path))
    #    if img_array.shape[0] > max_height:
    #        max_height = img_array.shape[0]
    #    if img_array.shape[1] > max_width:
    #        max_width = img_array.shape[1]

    #print(max_height)
    #print(max_width)

    # .npy
    #img_array = np.load(file_path)
    # .png
    img_array = np.asarray(Image.open(file_path))
    #twod = img_array[:, :, 0]

    #shape = img_array.shape
    #print(shape)
    # data must be 2d
    np.savetxt('out.txt', img_array, delimiter=',', fmt='%.0f')

def save_npy_to_txt(file_path):

    # .npy
    img_array = np.load(file_path)
    #twod = img_array[:, :, 0]

    #shape = img_array.shape
    #print(shape)
    # data must be 2d
    np.savetxt('out.txt', img_array, delimiter=',', fmt='%.0f')

def test():

    max_height = 500
    max_width = 500
    full_name = "/opt/fr/other/tools/VOC2012/SegmentationClass/2011_003103.png"
    img_array = np.asarray(Image.open(full_name))
    np.savetxt('/home/ShareFile/1.txt', img_array, delimiter=',', fmt='%.0f')
    file_name, ext = os.path.splitext(os.path.basename(full_name))
    if img_array.shape[0] < max_height:
        img_array = np.vstack([img_array, np.zeros([max_height - img_array.shape[0], img_array.shape[1]])])
    if img_array.shape[1] < max_width:
        img_array = np.hstack([img_array, np.zeros([img_array.shape[0], max_width - img_array.shape[1]])])
    np.savetxt('/home/ShareFile/2.txt', img_array, delimiter=',', fmt='%.0f')
    new_file = "/home/ShareFile/" + file_name + ".png"
    im = Image.fromarray(np.uint8(img_array))
    im.save(new_file)

def save_npy():

    #file_path = Path("/opt/fr/other/tools/VOC2012/SegmentationClass")
    file_path = Path("/opt/homepage-miraimon/public/segup/acreage/png")
    max_height = 500
    max_width = 500
    for i, image_file in enumerate(file_path.glob('**/*.png')):
        full_name = str(image_file)
        print(full_name)
        img_array = np.asarray(Image.open(image_file))
        file_name, ext = os.path.splitext(os.path.basename(full_name))
        if img_array.shape[0] < max_height:
            img_array = np.vstack([img_array, np.zeros([max_height - img_array.shape[0], img_array.shape[1]])])
        if img_array.shape[1] < max_width:
            img_array = np.hstack([img_array, np.zeros([img_array.shape[0], max_width - img_array.shape[1]])])

        img_array = to_towdense(img_array)

        new_file = "/home/ai/Datasets/acreage/" + file_name + ".npy"
        np.save(new_file, np.uint8(img_array))

def to_towdense(img_array):

    shape = img_array.shape
    dist_img = np.zeros([shape[0],shape[1]])
    
    height = shape[0]
    width = shape[1]
    for h in range(height):
        for w in range(width):
            dist_img[h,w] = img_array[h,w][0]

    return dist_img

def check(fpath):
    file_path = Path(fpath)
    for i, npy_file in enumerate(file_path.glob('**/*.npy')):
        npy_array = np.load(npy_file)
        print(npy_array[1][1])
        print(npy_array.shape)

if __name__ == '__main__':
    #resize_jpg()
    save_npy()
    #check("/home/ai/Datasets/acreage/")
    #check("/home/ai/Datasets/resizedjpg/")
    #save_npy_to_txt("/home/ai/Datasets/acreage/b1dab7c5-cccc-4bba-bbd1-9e8b7cbe40e2.npy")

