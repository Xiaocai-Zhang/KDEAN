# code for KDEAN
def warn(*args,**kwargs):
    pass
import warnings
warnings.warn=warn
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import tensorflow as tf
from tensorflow.keras import optimizers, callbacks
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras import backend as K
import numpy as np
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix,matthews_corrcoef,roc_auc_score
import argparse
from datetime import datetime




parser = argparse.ArgumentParser()
parser.add_argument('--train', type = bool, default = False, help = 'whether to train model')
parser.add_argument('--dataset', type = str, default = "Homo Sapiens", help = 'specific dataset')
parser.add_argument('--batchsize', type = int, default = 128, help = 'batch size')
parser.add_argument('--epoch', type = int, default = 100, help = 'training epoch')
parser.add_argument('--layer', type = int, default = 3, help = 'number of layer')
parser.add_argument('--channel', type = int, default = 10, help = 'number of channel')
parser.add_argument('--kernel', type = int, default = 5, help = 'kernel size')
parser.add_argument('--hidden', type = int, default = 100, help = 'hidden unit')
parser.add_argument('--lr', type = float, default = 0.001, help = 'learning rate')
parser.add_argument('--lr_factor', type = float, default = 0.1, help = 'learning rate decay factor')
parser.add_argument('--patience', type = int, default = 5, help = 'patience for lr decay')
parser.add_argument('--dropout', type = int, default = 0.2, help = 'dropout rate')
parser.add_argument('--maxlensequence', type = int, default = 79, help = 'maximum length of peptide')
parser.add_argument('--train_sample', type = int, default = 72000, help = 'size of training data sample')
parser.add_argument('--train_val_ratio', type = float, default = 0.9, help = 'percenatge of training data')
parser.add_argument('--seed', type = int, default = 109, help = 'random seed')
parser.add_argument('--GPU', type = bool, default = False, help = 'whether to use GPU acceleration')


def KnowledgeFeature(sequence):
    '''
    to extract the knowledge info. from peptide sequence
    :param sequence: peptide sequence
    :return: knowldege info.
    '''
    res = [0 for _ in range(22)]
    for item in sequence:
        res[item-1] = res[item-1]+1
    return res


def PepCoding(file, max_len):
    '''
    encoding of peptide sequence
    :param file: peptide data file
    :param max_len: maximum sequence length
    :return: encoded vector
    '''
    aa_dict = {'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
               'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'S': 16, 'U': 17, 'T': 18,
               'W': 19, 'Y': 20, 'V': 21, 'X': 22}
    with open(file, 'r') as inf:
        lines = inf.read().splitlines()

    long_pep_counter = 0
    pep_codes = []
    knowledge_li = []
    labels = []
    for pep in lines:
        pep, label = pep.split(",")
        labels.append(int(label))
        if not len(pep) > max_len:
            current_pep = []
            for aa in pep:
                current_pep.append(aa_dict[aa])
            pep_codes.append(current_pep)
            knowledge_code = KnowledgeFeature(current_pep)
            knowledge_li.append(knowledge_code)
        else:
            long_pep_counter += 1
    data = tf.keras.utils.pad_sequences(pep_codes,padding='post')
    return data, np.array(knowledge_li), tf.convert_to_tensor(labels)


class TempoConvNetworks:
    '''
    Temporal convolution network
    '''
    def TcnBlock(self, x, dilation_rate, nb_filters, kernel_size, dropout, padding, layer=1):
        '''
        TCN block
        '''
        init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
        assert padding in ['causal', 'same']
        conv1 = Conv1D(filters=nb_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding=padding,
                       kernel_initializer=init)
        batch1 = BatchNormalization(axis=-1)
        ac1 = Activation('relu')

        drop1 = GaussianDropout(dropout)

        conv2 = Conv1D(filters=nb_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding=padding,
                       kernel_initializer=init)
        batch2 = BatchNormalization(axis=-1)
        ac2 = Activation('relu')

        drop2 = GaussianDropout(dropout)

        downsample = Conv1D(filters=nb_filters, kernel_size=1, padding='same', kernel_initializer=init)
        ac3 = Activation('relu')

        pre_x = x

        x = conv1(x)
        x = batch1(x)
        x = ac1(x)
        if args.train:
            x = drop1(x)
        x = conv2(x)
        x = batch2(x)
        x = ac2(x)
        if args.train:
            x = drop2(x)

        if pre_x.shape[-1] != x.shape[-1]:  # to match the dimensions
            pre_x = downsample(pre_x)

        assert pre_x.shape[-1] == x.shape[-1]

        try:
            out = ac3(pre_x + x)
        except:
            pre_x = tf.cast(pre_x,dtype=tf.float16)
            out = ac3(pre_x + x)

        return out

    def TcnNet(self, input, num_channels, kernel_size, dropout):
        assert isinstance(num_channels, list)
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_rate = 2 ** i
            input = self.TcnBlock(input, dilation_rate, num_channels[i], kernel_size, dropout=dropout, padding='causal', layer=i+1)

        out = input

        return out


class ExternalAttention(layers.Layer):
    '''
    external attention
    '''
    def __init__(self, d_model, S=64, **kwargs):
        super(ExternalAttention, self).__init__()
        self.mk = layers.Dense(S, use_bias=False)
        self.mv = layers.Dense(d_model, use_bias=False)

    def call(self, queries):
        attn = self.mk(queries)  # bs,n,S
        attn = tf.nn.softmax(attn, axis=1)  # bs,n,S
        attn = attn / tf.reduce_sum(attn, axis=2, keepdims=True)  # bs,n,S (l1_norm)
        out = self.mv(attn)  # bs,n,d_model

        return out


def train(train_inp,train_know,train_oup,val_inp,val_know,val_oup,num_model):
    '''
    network training
    '''
    num_channels = [args.channel] * args.layer

    # build deep learning model
    inp_shape = (args.maxlensequence,)
    input_1 = Input(shape=inp_shape)
    input_2 = Input(shape=22)

    embedding = Embedding(input_dim=23,
                       output_dim=10,
                       input_length=args.maxlensequence,
                       mask_zero=True)

    output_1 = embedding(input_1)
    output_2 = tf.reverse(output_1,axis=[1])

    # TCN block 1
    TCNetworks_1 = TempoConvNetworks()
    output_1 = TCNetworks_1.TcnNet(output_1, num_channels, args.kernel, args.dropout)
    output_1 = Bidirectional(LSTM(args.hidden, return_sequences=True))(output_1)
    output_1 = ExternalAttention(d_model=128, S=8)(output_1)

    # TCN block 2
    TCNetworks_2 = TempoConvNetworks()
    output_2 = TCNetworks_2.TcnNet(output_2, num_channels, args.kernel, args.dropout)
    output_2 = Bidirectional(LSTM(args.hidden, return_sequences=True))(output_2)
    output_2 = ExternalAttention(d_model=128, S=8)(output_2)

    # knowldege learning block
    output_3 = Dense(1024)(input_2)
    output_3 = Activation('relu')(output_3)
    output_3 = Dense(128)(output_3)
    output_3 = Activation('relu')(output_3)

    output = output_1+output_2
    output = K.sum(output, axis=1)
    output = output+output_3

    # desne layer
    output = Dense(1)(output)
    output = Activation('sigmoid')(output)

    model = Model(inputs=[input_1,input_2], outputs=output)
    opt = optimizers.Adam(learning_rate=args.lr)
    model.compile(loss='log_cosh', optimizer=opt)

    SaveModlFile = './Model/'+args.dataset+'/model_'+str(num_model)+'.h5'
    mcp_save = callbacks.ModelCheckpoint(SaveModlFile, save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=args.lr_factor, patience=args.patience,
                                                 min_lr=0.0001,
                                                 mode='min')

    model.fit(x=[train_inp,train_know], y=train_oup, epochs=args.epoch,
              batch_size=args.batchsize, validation_data=([val_inp,val_know], val_oup),
              callbacks=[mcp_save, reduce_lr_loss],
              verbose=0)

    return None


def evaluate_accuracy(prediction, grundtruth):
    '''
    performance evauation
    :param prediction: prediction prob distribution
    :param grundtruth: grundtruth prob distribution
    :return: accuracy, f1-score, MCC and AUC
    '''
    prediction_idx = np.round(prediction)
    grundtruth = np.expand_dims(grundtruth, axis=1)
    auc = roc_auc_score(grundtruth, prediction, average='macro')
    mcc = matthews_corrcoef(grundtruth, prediction_idx)
    conf_matx = confusion_matrix(grundtruth, prediction_idx)
    tn, fp, fn, tp = conf_matx.ravel()
    acc = (tp+tn)/(tp+tn+fp+fn)
    fscore = 2*tp/(2*tp+fp+fn)
    return acc,fscore,mcc,auc


def test(test_inp,test_know,test_oup,num_model):
    '''
    test on the test data
    '''
    SaveModlFile = './Model/'+args.dataset+'/model_'+str(num_model)+'.h5'
    model = load_model(SaveModlFile,custom_objects={'ExternalAttention':ExternalAttention})
    predictions_test = model.predict([test_inp,test_know], batch_size=args.batchsize,verbose=0)
    acc,fscore,mcc,auc = evaluate_accuracy(predictions_test, test_oup)
    results = f'\tmodel: {num_model} ,acc: {acc:.4f}, fscore: {fscore:.4f}, mcc: {mcc:.4f}, auc: {auc:.4f}'
    print(results)
    return acc,fscore,mcc,auc


def Norm(knowledge_data):
    '''
    knowldege normalization
    '''
    f_max = knowledge_data.max()
    f_min = knowledge_data.min()
    knowledge_data = (knowledge_data-f_min)/(f_max-f_min)
    return knowledge_data


if __name__ == '__main__':
    start = datetime.now()
    args = parser.parse_args()

    if args.GPU:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if args.dataset=='Homo Sapiens':
        args.maxlensequence = 79
        data, knowledge_data, label = PepCoding("./dataset/Homo Sapiens.csv", args.maxlensequence)
        args.train_sample = 72000
    elif args.dataset=='Mus Musculus':
        args.maxlensequence = 66
        data, knowledge_data, label = PepCoding("./dataset/Mus Musculus.csv", args.maxlensequence)
        args.train_sample = 72000
    elif args.dataset=='Trypsin Human':
        args.maxlensequence = 51
        data, knowledge_data, label = PepCoding("./dataset/Trypsin Human.csv", args.maxlensequence)
        args.train_sample = 84806
    elif args.dataset=='LysC Human':
        args.maxlensequence = 54
        data, knowledge_data, label = PepCoding("./dataset/LysC Human.csv", args.maxlensequence)
        args.train_sample = 45360
    else:
        raise TypeError("Dataset not found")

    knowledge_data = Norm(knowledge_data)
    knowledge_data = tf.convert_to_tensor(knowledge_data)

    acc_li, fscore_li, mcc_li, auc_li = [],[],[],[]
    for num_model in range(10):
        seed = args.seed+num_model
        tf.random.set_seed(seed)
        data_ = tf.random.shuffle(data)
        tf.random.set_seed(seed)
        knowledge_data_ = tf.random.shuffle(knowledge_data)
        tf.random.set_seed(seed)
        label_ = tf.random.shuffle(label)

        train_val_data = data_[:args.train_sample]
        train_val_knowledge = knowledge_data_[:args.train_sample]
        train_val_label = label_[:args.train_sample]

        test_data = data_[args.train_sample:]
        test_knowledge = knowledge_data_[args.train_sample:]
        test_label = label_[args.train_sample:]

        train_data = train_val_data[:int(train_val_data.shape[0]*args.train_val_ratio)]
        train_knowledge = train_val_knowledge[:int(train_val_data.shape[0] * args.train_val_ratio)]
        train_label = train_val_label[:int(train_val_data.shape[0]*args.train_val_ratio)]

        val_data = train_val_data[int(train_val_data.shape[0]*args.train_val_ratio):]
        val_knowledge = train_val_knowledge[int(train_val_data.shape[0] * args.train_val_ratio):]
        val_label = train_val_label[int(train_val_data.shape[0]*args.train_val_ratio):]

        if args.train:
            print('Training model %s'%num_model)
            train(train_data, train_knowledge, train_label, val_data, val_knowledge,val_label,num_model)

        acc,fscore,mcc,auc = test(test_data,test_knowledge,test_label,num_model)
        acc_li.append(acc)
        fscore_li.append(fscore)
        mcc_li.append(mcc)
        auc_li.append(auc)

    df = pd.DataFrame({'acc':acc_li,'fscore':fscore_li,'mcc':mcc_li,'auc':auc_li})
    print('##########################')
    print('Mean')
    print(df.mean())
    print('Standard Deviation')
    print(df.std())
    duration = (datetime.now() - start).total_seconds() / 3600
    print('##########################')
    print("computational time: %s h" % (duration))
