import numpy as np
import sys
import pickle
import random
from PIL import Image
import cv2
from keras.utils import np_utils
import config
from sklearn.metrics import classification_report
from keras import backend as K
import math

server = config.server()
data_output_path = config.data_output_path()
data_folder_seq = r'{}seq3/'.format(data_output_path)

def getTrainData(keys,batch_size,dataset,classes,train,data_type): 
    """
    mode 1: Single Stream
    mode 2: Two Stream
    mode 3: Multiple Stream
    """
    mode = len(data_type)
    while 1:
        for i in range(0, len(keys), batch_size):

            if mode == 1:
                X_train, Y_train = stack_single_sequence(
                    chunk=keys[i:i+batch_size],
                    data_type=data_type,
                    dataset=dataset,
                    train=train)
            else:
                X_train, Y_train=stack_multi_sequence(
                    chunk=keys[i:i+batch_size],
                    multi_data_type=data_type,
                    dataset=dataset,
                    train=train)

            Y_train = np_utils.to_categorical(Y_train,classes)
            if train != 'train':
                print 'Test batch {}'.format(i/batch_size+1)
            yield X_train, np.array(Y_train)

def stack_seq_rgb(path_video,render_rgb,dataset,train):
    return_stack = []
    data_folder_rgb = r'{}{}-rgb-3/'.format(data_output_path,dataset)

    size = random_size()
    mode_crop = random.randint(0, 1)
    flip = random.randint(0, 1)
    mode_corner_crop = random.randint(0, 4)
    x = random.randint(0, 340-size)
    y = random.randint(0, 256-size)

    if train == 'valid':
        size = 224

    for i in render_rgb:
        rgb = cv2.imread(data_folder_rgb + path_video + '/' + str(i+10) + '.jpg')
        if rgb is None:
            print data_folder_rgb + path_video + '/' + str(i+10) + '.jpg'
            sys.exit()
        if train == 'train':
            rgb = random_crop(rgb, size, mode_crop, mode_corner_crop, x, y)
            rgb = random_flip(rgb, size, flip)
        else:
            rgb = image_crop(rgb, (340-size)/2, (256-size)/2, size)

        height, width, channel = rgb.shape
        if height == size:
            if size != 224:
                rgb = cv2.resize(rgb, (224, 224))
            print size
            rgb = rgb.astype('float16',copy=False)
            rgb/=255
            rgb_nor = rgb - rgb.mean()
        else:
            print(mode_crop, flip, mode_corner_crop, size, height, x, y)
            sys.exit()

        return_stack.append(rgb_nor)
    return np.array(return_stack)

def stack_seq_optical_flow(path_video,render_opt,data_type,dataset,train):

    arrays = []
    return_data = []
    len_render_opt = len(render_opt)

    for k in range(len_render_opt):
        for i in range(k*20 + 0, k*20 + 20):
            img = cv2.imread(data_folder_seq + path_video + '/opt' + str(data_type) + '-' + str(i) + '.jpg', 0)
            if img is None:
                print 'Error render optical flow'
                sys.exit()
            height, width = img.shape
            crop_pos = int((width-height)/2)
            img = img[:,crop_pos:crop_pos+height]
            resize_img = cv2.resize(img, (224, 224))

            resize_img = resize_img.astype('float16',copy=False)
            resize_img/=255
            opt_nor = resize_img - resize_img.mean()

            arrays.append(opt_nor)

        nstack = np.dstack(arrays)
        arrays = []
        return_data.append(nstack)

    if (len_render_opt == 1):
        return_data.append(nstack)
        return_data.append(nstack)

    return (return_data)

def stack_single_sequence(chunk,data_type,dataset,train):
    labels = []
    stack_return = []
    if data_type[0] == 0:
        for rgb in chunk:
            labels.append(rgb[2])
            stack_return.append(stack_seq_rgb(rgb[0],rgb[1],dataset,train))
    else:
        for opt in chunk:
            labels.append(opt[2])
            stack_return.append(stack_seq_optical_flow(opt[0],opt[1],data_type[0],dataset,train))

    if len(stack_return) < len(chunk):
        print 'Stacked data error'
        sys.exit()

    return np.array(stack_return), labels

def stack_multi_sequence(chunk,multi_data_type,dataset):
    labels = []
    returns = []
    stack_return = []

    for data_type in multi_data_type:
        stack_return.append([])

    for sample in chunk:
        labels.append(sample[2])

        s = 0
        for data_type in multi_data_type:
            if data_type == 0:
                stack_return[s].append(stack_seq_rgb(sample[0],sample[1],dataset))
            else:
                stack_return[s].append(stack_seq_optical_flow(sample[0],sample[1],data_type,dataset))
            s+=1

    if len(stack_return[0]) < len(chunk):
        print 'Stacked data error'
        sys.exit()

    for i in range(len(multi_data_type)):
        returns.append(np.array(stack_return[i]))

    return returns, labels


def getClassData(keys,cut=0):
    labels = []
    if cut == 0:
        for opt in keys:
            labels.append(opt[2])
    else:
        i = 0
        for opt in keys:
            labels.append(opt[2])
            i += 1
            if i >= cut:
                break

    return labels

def getScorePerVideo(result, data):
    indVideo = []
    dataVideo = []
    length = len(data)
    for i in range(length):
        name = data[i][0].split('/')[1]
        if name not in indVideo:
            indVideo.append(name)
            dataVideo.append([name,data[i][2],result[i], 1])
        else:
            index = indVideo.index(name)
            dataVideo[index][2] = dataVideo[index][2] + result[i]
            dataVideo[index][3] += 1

    resultVideo = []
    classVideo = []
    len_data = len(dataVideo)
    for i in range(len_data):
        pred = dataVideo[i][2] / dataVideo[i][3]
        resultVideo.append(pred)
        classVideo.append(dataVideo[i][1])

    resultVideoArr = np.array(resultVideo)
    classVideoArr = np.array(classVideo)

    y_classes = resultVideoArr.argmax(axis=-1)
    return (classification_report(classVideoArr, y_classes, digits=6))

def convert_weights(weights, depth, size=3, ins=32):
    mat = weights[0]
    mat2 = np.empty([size,size,depth,ins])
    for i in range(ins):
        x=(mat[:,:,0,i] + mat[:,:,1,i] + mat[:,:,2,i])/3
        for j in range(depth):
            mat2[:,:,j,i] = x
    return [mat2]

def random_size():
    size = [256,224,192,168]
    return random.choice(size)

def random_flip(image, size, flip):
    image_flip = image.copy()
    if (flip==1):
        image_flip = cv2.flip(image_flip, 1)
    return image_flip

def random_crop(image, size, mode_crop, mode_corner_crop, x, y):
    if mode_crop == 0:
        return random_corner_crop(image, size, mode_corner_crop)
    else:
        return image_crop(image, x, y, size)

def random_corner_crop(image, size, mode_corner_crop):
    if mode_corner_crop == 0:
        return image_crop(image, 0, 0, size)
    elif mode_corner_crop == 1:
        return image_crop(image, 340-size, 0, size)
    elif mode_corner_crop == 2:
        return image_crop(image, 0, 256-size, size)
    elif mode_corner_crop == 3:
        return image_crop(image, 340-size, 256-size, size)
    else:
        return image_crop(image, (340-size)/2, (256-size)/2, size)
       
def image_crop(image, x, y, size):
    return image[y:y+size,x:x+size,:]