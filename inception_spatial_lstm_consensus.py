import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--process', help='Process', default='train')
parser.add_argument('-data', '--dataset', help='Dataset', default='ucf101')
parser.add_argument('-b', '--batch', help='Batch size', default=16, type=int)
parser.add_argument('-c', '--classes', help='Number of classes', default=101, type=int)
parser.add_argument('-e', '--epoch', help='Number of epochs', default=5, type=int)
parser.add_argument('-r', '--retrain', help='Number of old epochs when retrain', default=0, type=int)
parser.add_argument('-cross', '--cross', help='Cross fold', default=1, type=int)
parser.add_argument('-s', '--summary', help='Show model', default=0, type=int)
parser.add_argument('-lr', '--lr', help='Learning rate', default=1e-4, type=float)
parser.add_argument('-decay', '--decay', help='Decay', default=0.0, type=float)
parser.add_argument('-fine', '--fine', help='Fine-tuning', default=1, type=int)
args = parser.parse_args()
print args

import sys
import config
import models
from keras import optimizers
from keras.applications.inception_v3 import InceptionV3
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, ZeroPadding2D
from keras.layers import TimeDistributed, Activation, AveragePooling1D, LSTM

process = args.process
if process == 'train':
    train = True
    retrain = False
    old_epochs = 0
elif process == 'retrain':
    train = True
    retrain = True
    old_epochs = args.retrain
else:
    train = False
    retrain = False

batch_size = args.batch
classes = args.classes
epochs = args.epoch
cross_index = args.cross
dataset = args.dataset
pre_file = 'incept_spatial_lstm_consensus'

seq_len = 3
n_neurons = 256
dropout = 0.5

if train & (not retrain):
    weights = 'imagenet'
else:
    weights = None
if args.fine == 1:
    fine = True
else:
    fine = False
inception = InceptionV3(
    input_shape=(224,224,3),
    pooling='avg',
    include_top=False,
    weights=weights,
)
result_model = Sequential()
result_model.add(TimeDistributed(inception, input_shape=(seq_len, 224,224,3)))
result_model.add(LSTM(n_neurons, return_sequences=True))
result_model.add(Flatten())
result_model.add(Dropout(dropout))
result_model.add(Dense(classes, activation='softmax'))

if (args.summary == 1):
    result_model.summary()
    sys.exit()

lr = args.lr 
decay = args.decay

if train:
    if not retrain:
        for layer in inception.layers:
            layer.trainable = False
        result_model.summary()
        result_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=["accuracy"])
        models.train_process(result_model, pre_file, data_type=[0], epochs=3, dataset=dataset,
            retrain=retrain,  classes=classes, cross_index=cross_index, 
            seq_len=seq_len, old_epochs=0, batch_size=batch_size, fine=False)
        old_epochs = 3
    
    # Retrain without preeze some layers
    if fine:
        for layer in inception.layers[:172]:
            layer.trainable = False
        for layer in inception.layers[172:]:
            layer.trainable = True
        inception.get_layer('batch_normalization_1').trainable = True
    result_model.summary()
    result_model.compile(loss='categorical_crossentropy',
                     optimizer=optimizers.SGD(lr=lr, decay=decay, momentum=0.9, nesterov=True),
                     metrics=['accuracy'])
    models.train_process(result_model, pre_file, data_type=[0], epochs=epochs, dataset=dataset,
        retrain=retrain,  classes=classes, cross_index=cross_index, 
        seq_len=seq_len, old_epochs=old_epochs, batch_size=batch_size, fine=False)

else:
    models.test_process(result_model, pre_file, data_type=[0], epochs=epochs, dataset=dataset,
        classes=classes, cross_index=cross_index,
        seq_len=seq_len, batch_size=batch_size)
    
