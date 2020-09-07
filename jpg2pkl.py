import os
import numpy as np
import cv2
import sys
import glob
import pickle
from multiprocessing import Pool
import pandas as pd

def convert_csv_to_list(csv_path):
    data = pd.read_csv(csv_path, delimiter=' ', header=None)
    keys = []
    for i in range(data.shape[0]):
        row = data.iloc[i, :]
        slash_rows = data.iloc[i, 0].split('/')
        class_name = slash_rows[0]
        basename = slash_rows[1].split('.')[0]
        keys.append(basename)
        print('row',row)
        print('basename',basename)
    return keys

train_txt = 'work/split_ucf/trainlist01.txt'
test_txt = 'work/split_ucf/testlist01.txt'
train_list = convert_csv_to_list(train_txt)
test_list = convert_csv_to_list(test_txt)


label_dic = np.load('label_dir.npy', allow_pickle=True).item()
print(label_dic)

source_dir = 'work/jpg_video'
target_train_dir = 'work/train'
target_test_dir = 'work/test'
if not os.path.exists(target_train_dir):
    os.mkdir(target_train_dir)
if not os.path.exists(target_test_dir):
    os.mkdir(target_test_dir)

for key in label_dic:
    # key Rowing
    each_mulu = key + '_jpg'  #  Rowing_jpg

    label_dir = os.path.join(source_dir, each_mulu)
    label_mulu = os.listdir(label_dir)
    tag = 1
    for each_label_mulu in label_mulu:
        # each_label_mulu  v_Rowing_g25_c02
        image_file = os.listdir(os.path.join(label_dir, each_label_mulu))
        image_file.sort()
        image_name = image_file[0][:-6]  # image_file[0]  v_Rowing_g25_c02_1.jpg
        image_num = len(image_file)
        frame = []
        vid = image_name                # vid - image_name - v_Rowing_g25_c02
        for i in range(image_num):
            image_path = os.path.join(os.path.join(label_dir, each_label_mulu), image_name + '_' + str(i+1) + '.jpg')
            frame.append(image_path)

        output_pkl = vid + '.pkl'
        if each_label_mulu in train_list:
            output_pkl = os.path.join(target_train_dir, output_pkl)
        if each_label_mulu in test_list:
            output_pkl = os.path.join(target_test_dir, output_pkl)
        # if tag < 40:
        #     output_pkl = os.path.join(target_train_dir, output_pkl)
        # elif tag >= 40 and tag<45:
        #     output_pkl = os.path.join(target_test_dir, output_pkl)
        # else:
        #     output_pkl = os.path.join(target_val_dir, output_pkl)
        tag += 1
        f = open(output_pkl, 'wb')
        pickle.dump((vid, label_dic[key], frame), f, -1)
        f.close()
        # print("frame",frame)
        # print("image_name",image_name)
train_pkls = os.listdir('work/train')
test_pkls = os.listdir('work/test')
print("train_list",len(train_list),len(train_pkls))
print("test_list",len(test_list),len(test_pkls))