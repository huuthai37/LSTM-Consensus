import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-data', '--dataset', help='Dataset', default='ucf101')
parser.add_argument('-stream', '--stream', help='Stream', default='spatial')
parser.add_argument('-sample', '--sample', help='Batch size', default=1, type=int)
parser.add_argument('-cross', '--cross', help='Cross fold', default=1, type=int)
args = parser.parse_args()
print args

import sys
import config
import models


batch_size = 1
classes = 101
cross_index = args.cross
dataset = args.dataset

seq_len = 3
n_neurons = 256
dropout = 0.8
stream = args.stream

index = args.sample

if stream == 'spatial':
    pre_file = 'incept229_spatial_lstm{}'.format(n_neurons)

    result_model = models.InceptionSpatialLSTMConsensus(
                        n_neurons=n_neurons, seq_len=seq_len, classes=classes, 
                        weights=None, dropout=dropout, fine=True, retrain=False,
                        pre_file=pre_file,old_epochs=0,cross_index=cross_index)

    model.load_weights('weights-old/save-imp/{}_{}e_cr{}.h5'.format(pre_file,45,cross_index))
    data_type = [0]

else:
    print "Error stream"

result_model.compile(loss='categorical_crossentropy',
                     optimizer='sgd',
                     metrics=['accuracy'])

out_file = r'{}database/{}-test{}-split{}.pickle'.format(config.data_output_path(),dataset,seq_len,cross_index)
with open(out_file,'rb') as f2:
    keys = pickle.load(f2)

if index > (len(keys) - 1):
    print 'Out of number data test'
    sys.exit()

class_file = 'data/{}-classInd.txt'.format(dataset)
classInd=[]
with open(class_file) as f0:
    for line in f0:
        class_name = line.rstrip()
        if class_name:
            classInd.append(class_name)

y_pred = model.predict_generator(
    gd.getTrainData(
        keys=keys[index],batch_size=1,dataset=dataset,classes=classes,train='test',data_type=data_type,split_sequence=False), 
    max_queue_size=1, 
    steps=1)

classInd2 = [x for _,x in sorted(zip(y_pred,classInd),reverse=True)]
y_pred = sorted(y_pred,reverse=True)

print ('True class:', classInd[keys[index][2]])
for i in range(5):
    print(classInd2[i], y_pred[i])     




    


    