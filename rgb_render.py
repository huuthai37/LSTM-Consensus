import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', help='Dataset', default='ucf101')
parser.add_argument('-s', '--segment', help='Number of segments', default=3, type=int)
parser.add_argument('-debug', '--debug', help='Number of classes', default=1, type=int)
args = parser.parse_args()
print args

import cv2
import os
import sys
import random
import numpy as np
import config
import pickle

# Lay tuy chon 
dataset = args.dataset
num_seq = args.segment
if args.debug == 1:
    debug = True
else:
    debug = False

def count_frames(path):
    cap = cv2.VideoCapture(path)
    i = 0
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break  
        i += 1
    cap.release()
    return i

# Cau hinh folder du lieu
server = config.server()
data_input_folder = config.data_input_path()
output_path = config.data_output_path()

class_file = r'data/{}-classInd.txt'.format(dataset)
text_file = r'data/{}-datalist.txt'.format(dataset)
out_file_folder = r'{}database/'.format(output_path)
out_file = r'{}data-{}-{}.pickle'.format(out_file_folder,dataset,num_seq)
data_output_folder = r'{}{}-rgb-{}/'.format(output_path,dataset,num_seq)

# Tao folder chinh
if not os.path.isdir(data_output_folder):
    os.makedirs(data_output_folder) # tao data_output_folder/
    print 'Create directory ' + data_output_folder

if not os.path.isdir(out_file_folder):
    os.makedirs(out_file_folder) # tao out_file_folder/
    print 'Create directory ' + out_file_folder

# Tao class index tu file classInd.txt
classInd=[]
with open(class_file) as f0:
    for line in f0:
        class_name = line.rstrip()
        if class_name:
            classInd.append(class_name)

count = 0
data = []

with open(text_file) as f:
    for line in f:
        arr_line = line.rstrip().split(' ')[0] # return folder/subfolder/name.mpg

        path_video = arr_line.split('/') # return array (folder,subfolder,name.mpg)
        num_name = len(path_video) # return 3
        name_video = path_video[num_name - 1].split('.')[0] # return name ko co .mpg
        folder_video = path_video[0] # return folder
        path = data_output_folder + folder_video + '/' #return data-output/folder/
        video_class = classInd.index(folder_video)

        # tao folder moi neu chua ton tai
        if not os.path.isdir(path):
            os.makedirs(path)
            print 'Created folder: {}'.format(folder_video)

        if not os.path.isdir(path + name_video):
            os.makedirs(path + name_video) #tao data-output/foldet/name/

        length = count_frames(data_input_folder + arr_line)
            # length = count_frames(data_input_folder + path_video[num_name - 1])
        
        if not debug:
            cap = cv2.VideoCapture(data_input_folder + arr_line)

        divide = length / num_seq
        train_render = []
        if length > 60:
            for i in range(num_seq):
                if i < num_seq - 1:
                    k = np.random.randint(divide*i,divide*(i+1)-19)
                else:
                    k = np.random.randint(divide*i,length-20)
                train_render.append(k)
        else:
            if (length > 30):
                train_render = [0, length/2-10, length-21]
            elif (length <= 30) & (length > 20):
                train_render = [-10, length/2-10, length-11]
                print('Short video', train_render)
            else:
                print 'Skipped'
                continue

        data.append([folder_video + '/' + name_video, train_render, video_class, 0])

        test_render = []
        divide = (length - 1) * 1.0 / 24
        for i in range(25):
            test_render.append(int(round(i * divide)))

        os.chdir(path + name_video)

        i = -1
        if not debug:
            while(True):
                # Capture frame-by-frame
                ret, frame = cap.read()
                if not ret:
                    break
                i += 1  
                if ((i-10) in train_render) | (i in test_render):
                    resize_img = cv2.resize(frame, (340, 256))
                    cv2.imwrite('{}.jpg'.format(i),resize_img)  
            # Giai phong capture
            cap.release() 

        count += 1
        if (count % 100 == 0):
            print r'Created {} samples'.format(count)


print 'Generate {} samples for {} dataset'.format(len(data),dataset)

# Ghi du lieu dia chi ra file
with open(out_file,'wb') as f2:
    pickle.dump(data,f2)