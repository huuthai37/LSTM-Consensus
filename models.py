import config
import pickle
import random
import time
import numpy as np
import get_data as gd
import keras.backend as K
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, ZeroPadding2D, Concatenate
from keras.layers import TimeDistributed, Activation, AveragePooling1D
from keras.layers import LSTM, GlobalAveragePooling1D, Reshape, MaxPooling1D, Conv2D
from keras.layers import Input, Lambda, Average, average
from keras.applications.mobilenet import MobileNet
from keras.applications.inception_v3 import InceptionV3
from keras.applications.densenet import DenseNet169, DenseNet121, DenseNet201
from keras.applications.resnet50 import ResNet50
from inceptionv3 import TempInceptionV3
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from keras import optimizers

server = config.server()
data_output_path = config.data_output_path()

def relu6(x):
    return K.relu(x, max_value=6)

def SpatialLSTMConsensus(n_neurons=128, seq_len=3, classes=101, weights='imagenet', dropout=0.5):
    mobilenet = MobileNet(
        input_shape=(224,224,3),
        pooling='avg',
        include_top=False,
        weights=weights,
    )

    result_model = Sequential()
    result_model.add(TimeDistributed(mobilenet, input_shape=(seq_len, 224,224,3)))
    result_model.add(LSTM(n_neurons, return_sequences=True))
    result_model.add(Flatten())
    result_model.add(Dropout(dropout))
    result_model.add(Dense(classes, activation='softmax'))

    return result_model

def InceptionSpatialLSTMConsensus(n_neurons=128, seq_len=3, classes=101, weights='imagenet', 
    dropout=0.5, fine=True, retrain=False, pre_file='',old_epochs=0,cross_index=1):
    inception = InceptionV3(
        input_shape=(299,299,3),
        pooling='avg',
        include_top=False,
        weights=weights,
    )
    # for i, layer in enumerate(inception.layers):
    #     print(i, layer.name)
    # for layer in inception.layers[:173]:
    #     layer.trainable = False
    # for layer in inception.layers[173:]:
    #     layer.trainable = True
    # count = 0
    # for i, layer in enumerate(inception.layers):
    #     if layer.name == 'conv2d_1':
    #         layer.trainable = True
    #         print 'Unpreeze ' + layer.name
    #         continue
    #     a = layer.name.split('_')[0]
    #     if a == 'batch':
    #         layer.trainable = True
    #         count += 1
    # print 'Have ' + str(count) + ' BN layers'

    result_model = Sequential()
    result_model.add(TimeDistributed(inception, input_shape=(seq_len, 299,299,3)))
    result_model.add(LSTM(n_neurons, return_sequences=True))
    # result_model.add(AveragePooling1D(pool_size=seq_len))
    result_model.add(Flatten())
    result_model.add(Dropout(dropout))
    result_model.add(Dense(classes, activation='softmax'))

    if retrain:
        result_model.load_weights('weights/{}_{}e_cr{}.h5'.format(pre_file,old_epochs,cross_index))

    # if not fine:
    #     for layer in inception.layers:
    #         layer.trainable = True
    #     result_model.summary()

    return result_model

def InceptionTemporalLSTMConsensus(n_neurons=256, seq_len=3, classes=101, weights='imagenet', 
    dropout=0.8, fine=True, retrain=False, pre_file='',old_epochs=0,cross_index=1):
    inception = TempInceptionV3(
        input_shape=(None,None,20),
        pooling='avg',
        include_top=False,
        weights=weights,
        depth=20
    )

    result_model = Sequential()
    result_model.add(TimeDistributed(inception, input_shape=(seq_len, 299,299,20)))
    result_model.add(LSTM(n_neurons, return_sequences=True))
    result_model.add(Flatten())
    result_model.add(Dropout(dropout))
    result_model.add(Dense(classes, activation='softmax'))

    if retrain:
        result_model.load_weights('weights/{}_{}e_cr{}.h5'.format(pre_file,old_epochs,cross_index))

    return result_model

def InceptionMultiLSTMConsensus(n_neurons=256, seq_len=3, classes=101, weights='imagenet', 
    dropout=0.8, fine=True, retrain=False, pre_file='',old_epochs=0,cross_index=1, pre_train=None):

    if weights != 'imagenet':
        weight = None
    else:
        weight = weights

    spatial = InceptionSpatialLSTMConsensus(
                    n_neurons=n_neurons, seq_len=seq_len, classes=classes, 
                    weights=weight, dropout=dropout, fine=fine, retrain=False,
                    pre_file=pre_file,old_epochs=old_epochs,cross_index=cross_index)

    spatial.pop()
    spatial.pop()

    if (weights == 'pretrain') & (not retrain):
        spatial.load_weights('weights/incept229_spatial_lstm{}_{}e_cr{}.h5'.format(n_neurons,pre_train[0],cross_index))
        print 'load spatial weights'

    temporal = InceptionTemporalLSTMConsensus(
                    n_neurons=n_neurons, seq_len=seq_len, classes=classes, 
                    weights=weight, dropout=dropout, fine=fine, retrain=False,
                    pre_file=pre_file,old_epochs=old_epochs,cross_index=cross_index)

    if (weights == 'pretrain') & (not retrain):
        temporal.load_weights('weights/incept229_temporal_lstm{}_{}e_cr{}.h5'.format(n_neurons,pre_train[1],cross_index))
        print 'load temporal weights'

    temporal.pop()
    temporal.pop()

    concat = Concatenate()([spatial.output, temporal.output])
    concat = Dropout(dropout)(concat)
    concat = Dense(classes, activation='softmax')(concat)

    result_model = Model(inputs=[spatial.input, temporal.input], outputs=concat)

    if retrain:
        result_model.load_weights('weights/{}_{}e_cr{}.h5'.format(pre_file,old_epochs,cross_index))
        print 'load old weights'

    return result_model

def DenseNetSpatialLSTMConsensus(n_neurons=128, seq_len=3, classes=101, weights='imagenet', dropout=0.5, id=121):
    if id == 169:
        densenet = DenseNet169(
            input_shape=(224,224,3),
            pooling='avg',
            include_top=False,
            weights=weights,
        )
    elif id == 201:
        densenet = DenseNet201(
            input_shape=(224,224,3),
            pooling='avg',
            include_top=False,
            weights=weights,
        )
    else:
        densenet = DenseNet121(
            input_shape=(224,224,3),
            pooling='avg',
            include_top=False,
            weights=weights,
        )
    count = 0
    for i, layer in enumerate(densenet.layers):
        a = layer.name.split('_')
        if ('batch' in a) | ('bn' in a) :
            layer.trainable = True
            count += 1
    print 'Have ' + str(count) + ' BN layers'

    result_model = Sequential()
    result_model.add(TimeDistributed(densenet, input_shape=(seq_len, 224,224,3)))
    result_model.add(LSTM(n_neurons, return_sequences=True))
    result_model.add(Flatten())
    result_model.add(Dropout(dropout))
    result_model.add(Dense(classes, activation='softmax'))

    return result_model

def ResnetSpatialLSTMConsensus(n_neurons=256, seq_len=3, classes=101, weights='imagenet', 
    dropout=0.8, fine=True, retrain=False, pre_file='',old_epochs=0,cross_index=1):
    resnet = ResNet50(
        input_shape=(224,224,3),
        pooling='avg',
        include_top=False,
        weights=weights,
    )

    # for layer in resnet.layers[:175]:
    #     layer.trainable = False
    # for layer in resnet.layers[175:]:
    #     layer.trainable = True

    result_model = Sequential()
    result_model.add(TimeDistributed(resnet, input_shape=(seq_len, 224,224,3)))
    result_model.add(LSTM(n_neurons, return_sequences=True))
    result_model.add(Flatten())
    result_model.add(Dropout(dropout))
    result_model.add(Dense(classes, activation='softmax'))

    if retrain:
        result_model.load_weights('weights/{}_{}e_cr{}.h5'.format(pre_file,old_epochs,cross_index))

    if not fine:
        for layer in resnet.layers:
            layer.trainable = True
        result_model.summary()

    return result_model

def SpatialConsensus(seq_len=3, classes=101, weights='imagenet', dropout=0.5):
    mobilenet_no_top = MobileNet(
        input_shape=(224,224,3),
        pooling='avg',
        include_top=False,
        weights=weights,
    )
    # x = Reshape((1,1,1024), name='reshape_1')(mobilenet_no_top.output)
    x = Dropout(dropout, name='dropout')(mobilenet_no_top.output)
    # x = Conv2D(classes, (1, 1),
    #                padding='same', name='conv_preds')(x)
    # x = Activation('softmax', name='act_softmax')(x)
    # x = Reshape((classes,), name='reshape_2')(x)
    x = Dense(classes, activation='softmax')(x)
    mobilenet = Model(inputs=mobilenet_no_top.input, outputs=x)
    # mobilenet.summary()

    result_model = Sequential()
    result_model.add(TimeDistributed(mobilenet, input_shape=(seq_len, 224,224,3)))
    # result_model.add(AveragePooling1D(pool_size=seq_len))
    result_model.add(GlobalAveragePooling1D())
    # result_model.add(Dropout(dropout))
    # result_model.add(Flatten())
    # result_model.add(Activation('linear'))

    return result_model

def SpatialConsensus2(seq_len=3, classes=101, weights='imagenet', dropout=0.5):
    mobilenet_no_top = MobileNet(
        input_shape=(224,224,3),
        pooling='avg',
        include_top=False,
        weights=weights,
    )
    # x = Reshape((1,1,1024), name='reshape_1')(mobilenet_no_top.output)
    
    # x = Conv2D(classes, (1, 1),
    #                padding='same', name='conv_preds')(x)
    x = Dropout(dropout, name='dropout')(mobilenet_no_top.output)
    # x = Activation('softmax', name='act_softmax')(x)
    # x = Reshape((classes,), name='reshape_2')(x)
    x = Dense(classes, activation='softmax')(x)
    mobilenet = Model(inputs=mobilenet_no_top.input, outputs=x)
    # mobilenet.summary()

    input_1 = Input((224,224,3))
    input_2 = Input((224,224,3))
    input_3 = Input((224,224,3))

    y_1 = mobilenet(input_1)
    y_2 = mobilenet(input_2)
    y_3 = mobilenet(input_3)

    z = average([y_1, y_2, y_3])
    # z = Dropout(dropout, name='dropout')(z)
#     z = Activation('softmax')(z)

    result_model = Model(inputs=[input_1, input_2, input_3], outputs=z)

    return result_model

def TemporalLSTMConsensus(n_neurons=128, seq_len=3, classes=101, weights='imagenet', dropout=0.5, depth=20):
    mobilenet = mobilenet_remake(
        name='temporal',
        input_shape=(224,224,depth),
        classes=classes,
        weight=weights,
        depth=depth
    )

    result_model = Sequential()
    result_model.add(TimeDistributed(mobilenet, input_shape=(seq_len, 224,224,depth)))
    result_model.add(LSTM(n_neurons, return_sequences=True))
    result_model.add(Flatten())
    result_model.add(Dropout(dropout))
    result_model.add(Dense(classes, activation='softmax'))

    return result_model

def TemporalLSTMConsensus(n_neurons=128, seq_len=3, classes=101, weights='imagenet', dropout=0.5, depth=20):
    inception = inception_remake(
        name='temporal',
        input_shape=(224,224,depth),
        classes=classes,
        weight=weights,
        depth=depth
    )

    result_model = Sequential()
    result_model.add(TimeDistributed(inception, input_shape=(seq_len, 224,224,depth)))
    result_model.add(LSTM(n_neurons, return_sequences=True))
    result_model.add(Flatten())
    result_model.add(Dropout(dropout))
    result_model.add(Dense(classes, activation='softmax'))

    return result_model

def mobilenet_remake(name, input_shape, classes, weight=None, non_train=False, depth=20):
    
    mobilenet = MobileNet(
        input_shape=(224,224,3),
        pooling='avg',
        include_top=False,
        weights=weight,
    )
    name = name + '_'

    # Disassemble layers
    layers = [l for l in mobilenet.layers]

    new_input = Input(shape=input_shape)
    x = ZeroPadding2D(padding=(1, 1))(new_input)
    x = Conv2D(filters=32, 
              kernel_size=(3, 3),
              padding='valid',
              use_bias=False,
              name=name+'conv_new',
              strides=(2,2))(x)

    for i in range(3, len(layers)):
        layers[i].name = str(name) + layers[i].name
        x = layers[i](x)

    model = Model(inputs=new_input, outputs=x)
    if weight is not None:
        model.get_layer(name+'conv_new').set_weights(gd.convert_weights(layers[2].get_weights(), depth))

    return model

from keras.callbacks import LearningRateScheduler

def train_process(model, pre_file, data_type, epochs=20, dataset='ucf101', 
    retrain=False, classes=101, cross_index=1, seq_len=3, old_epochs=0, batch_size=16, split_sequence=False, fine=True):

    out_file = r'{}database/{}-train{}-split{}.pickle'.format(data_output_path,dataset,seq_len,cross_index)
    valid_file = r'{}database/{}-test{}-split{}.pickle'.format(data_output_path,dataset,seq_len,cross_index)

    with open(out_file,'rb') as f1:
        keys = pickle.load(f1)
    len_samples = len(keys)

    with open(valid_file,'rb') as f2:
        keys_valid = pickle.load(f2)
    len_valid = len(keys_valid)

    print('-'*40)
    print('{} training'.format(pre_file))
    print 'Number samples: {}'.format(len_samples)
    print 'Number valid: {}'.format(len_valid)
    print('-'*40)

    histories = []
    if server:
        steps = len_samples/batch_size
        validation_steps = int(np.ceil(len_valid*1.0/batch_size))
    else:
        steps = len_samples/batch_size
        validation_steps = int(np.ceil(len_valid*1.0/batch_size))
    
    for e in range(epochs):
        print('Epoch', e+1)
        print('-'*40)

        random.shuffle(keys)

        # def exp_decay(epoch, lr):
        #     print ('Index',epoch, e)
        #     if (e % 3 == 0) & (e != 0): 
        #         lr = lr * 0.9
        #     return lr

        # lrate = LearningRateScheduler(exp_decay, verbose=1)

        time_start = time.time()

        history = model.fit_generator(
            gd.getTrainData(
                keys=keys,batch_size=batch_size,dataset=dataset,classes=classes,train='train',data_type=data_type,split_sequence=split_sequence), 
            verbose=1, 
            max_queue_size=20, 
            steps_per_epoch=steps, 
            epochs=1,
            validation_data=gd.getTrainData(
                keys=keys_valid,batch_size=batch_size,dataset=dataset,classes=classes,train='valid',data_type=data_type,split_sequence=split_sequence),
            validation_steps=validation_steps,
            # callbacks=[lrate]
        )
        run_time = time.time() - time_start

        histories.append([
            history.history['acc'],
            history.history['val_acc'],
            history.history['loss'],
            history.history['val_loss'],
            run_time
        ])
        model.save_weights('weights/{}_{}e_cr{}.h5'.format(pre_file,old_epochs+1+e,cross_index))

        with open('histories/{}_{}_{}_{}e_cr{}'.format(pre_file,seq_len,old_epochs,epochs,cross_index), 'wb') as file_pi:
            pickle.dump(histories, file_pi)

def test_process(model, pre_file, data_type, epochs=20, dataset='ucf101', 
    classes=101, cross_index=1, seq_len=3, batch_size=16, split_sequence=False):

    model.load_weights('weights/{}_{}e_cr{}.h5'.format(pre_file,epochs,cross_index))

    out_file = r'{}database/{}-test{}-split{}.pickle'.format(data_output_path,dataset,seq_len,cross_index)
    with open(out_file,'rb') as f2:
        keys = pickle.load(f2)
    
    len_samples = len(keys)

    print('-'*40)
    print('{} testing'.format(pre_file))
    print 'Number samples: {}'.format(len_samples)
    print('-'*40)

    Y_test = gd.getClassData(keys)
    steps = int(np.ceil(len_samples*1.0/batch_size))

    time_start = time.time()

    y_pred = model.predict_generator(
        gd.getTrainData(
            keys=keys,batch_size=batch_size,dataset=dataset,classes=classes,train='test',data_type=data_type,split_sequence=split_sequence), 
        max_queue_size=20, 
        steps=steps)

    run_time = time.time() - time_start

    with open('results/{}_{}e_cr{}.pickle'.format(pre_file,epochs,cross_index),'wb') as fw3:
        pickle.dump([y_pred, Y_test],fw3)

    y_classes = y_pred.argmax(axis=-1)
    print(classification_report(Y_test, y_classes, digits=6))
    print 'Run time: {}'.format(run_time)
