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

# Cau hinh folder du lieu
server = config.server()
data_input_folder = config.data_input_path()
output_path = config.data_output_path()

out_file_folder = r'{}database/'.format(output_path)
data_file = r'{}data-{}-{}.pickle'.format(out_file_folder,dataset,num_seq)

count = 0
with open(data_file,'rb') as f1:
    data = pickle.load(f1)

length_data = len(data)
data_folder_opt = r'{}{}-opt/'.format(output_path,dataset)
data_folder_seq_opt = r'{}{}-seq-opt/'.format(output_path,dataset)

if not os.path.isdir(data_folder_seq_opt + 'u'):
    os.makedirs(data_folder_seq_opt + 'u') # tao data_folder_seq_opt + 'u'/
    print 'Create directory ' + data_folder_seq_opt + 'u'

if not os.path.isdir(data_folder_seq_opt + 'v'):
    os.makedirs(data_folder_seq_opt + 'v') # tao data_folder_seq_opt + 'v'/
    print 'Create directory ' + data_folder_seq_opt + 'v'

for l in range(length_data):
    path_video = data[l][0]
    render_opt = data[l][1]
    name_video = path_video.split('/')[1]
    u = data_folder_opt + 'u/' + name_video + '/frame'
    v = data_folder_opt + 'v/' + name_video + '/frame'

    if not os.path.isdir(data_folder_seq_opt + 'u/' + name_video):
        os.makedirs(data_folder_seq_opt + 'u/' + name_video) # tao data_folder_seq_opt + 'u/' + name_video/
        print 'Create directory ' + data_folder_seq_opt + 'u/' + name_video

    if not os.path.isdir(data_folder_seq_opt + 'v/' + name_video):
        os.makedirs(data_folder_seq_opt + 'v/' + name_video) # tao data_folder_seq_opt + 'v/' + name_video/
        print 'Create directory ' + data_folder_seq_opt + 'v/' + name_video

    return_data = []

    if (render_opt[0] >= 0):
        render = render_opt
    else:
        render = [render_opt[1]]
    len_render_opt = len(render)

    for k in range(len_render_opt):
        nstack_u = np.zeros((2560,340))
        nstack_v = np.zeros((2560,340))
        for i in range(10):
            img_u = cv2.imread(u + str(render[k] + 5 + i).zfill(6) + '.jpg', 0)
            img_v = cv2.imread(v + str(render[k] + 5 + i).zfill(6) + '.jpg', 0)
            
            # img_u = np.ones((240,320))
            # img_v = np.ones((240,320))

            if (img_u is None) | (img_v is None):
                print 'Error render optical flow'
                print(u + str(render[k] + 5 + i).zfill(6) + '.jpg')
                sys.exit()
            hh, ww = img_u.shape
            if (hh != 256) | (ww != 340):
                img_u = cv2.resize(img_u, (340, 256))
                img_v = cv2.resize(img_v, (340, 256))
            nstack_u[(256*i):(256*(i+1)),:] = img_u
            nstack_v[(256*i):(256*(i+1)),:] = img_v

        os.chdir(data_folder_seq_opt + 'u/' + name_video)
        cv2.imwrite('{}.jpg'.format(k),nstack_u)
        os.chdir(data_folder_seq_opt + 'v/' + name_video)
        cv2.imwrite('{}.jpg'.format(k),nstack_v)

    if l%1000 == 0:
        print l



