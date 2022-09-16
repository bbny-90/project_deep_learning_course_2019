#!/usr/bin/env python3
# TF2
import h5py
import os
import numpy as np
import urllib.request as url
import tarfile
import tensorflow as tf
from utils.data_extractor import generate_trainable_data

def download_data_and_extract(data_tag, main_dir='./data/'):
    """
        this functions download and extract "data_tag" if
        it is not already created in "main_dir".
        Inputs:
        data_tag: an string among {train, test, extra}
        main_dir: the directory you want to save your data
        Output:
            no return.
    """
    data_url_path = "http://ufldl.stanford.edu/housenumbers/"
    assert data_tag in {'train','test','extra'}, "the data tage {} is not defined".format(data_tag)
    print("Start downloading {} data.".format(data_tag))
    url.urlretrieve(data_url_path + data_tag + ".tar.gz", main_dir + data_tag + ".tar.gz")
    print("Complete downloading {} data.".format(data_tag))
    print("Start untaring {} data.".format(data_tag))
    tar = tarfile.open(main_dir + data_tag + ".tar.gz")
    names = tar.getnames()
    for name in names:
        tar.extract(name, main_dir)
    tar.close()
    print("Complete untaring {} data.".format(data_tag))

def get_y_model_from_y_plain(y_plain):
    """
        Input:
        y_plain: is the label provided by the raw data-set
        Output:
        y_model: is the required label for our model which has
                 to be in a specific format.
    """
    y_dummy = tf.keras.utils.to_categorical(y_plain)
    y_model = []
    for i in range(6):
        y_model.append(y_dummy[:,i])
    return y_model

def get_y_plain_from_y_model(y_model):
    """
        Input:
        y_model: is our model label (for example what it predicts)
                 which needs to be tranformed to the original raw
                 format to be compared with the grand truth values.
        Output:
        y_plain: the label similar to the raw date-set label 
    """
    y_plain = list()
    for i in range(6):
        y_plain.append(y_model[i].argmax())
    return y_plain

def get_data_xy_plain(image_size = 64, tag="train", path = "./data/train/digitStruct.mat"):
    """
        this funtion provides the data set. if it is already in the path defined as input function
        it will read and return input and output as x and y, respectively. Otherwise it will
        download the data and generate approporiate data for our CNN model.
        Inputs:
        image_size: image size
        tag: data tage. it can be {"train", "test", "extra"}
        path: where is the data tag or where do you want to save data tag
        Outputs:
        x: input imagaes for the CNN model
        y: lable for the CNN model
        
    """
    main_dir='./data/'
    if not os.path.exists(main_dir): os.mkdir(main_dir)
    if os.path.exists(path): data = h5py.File(path, 'r')
    else:
        download_data_and_extract(tag, main_dir)
        temp = main_dir + tag + "/digitStruct.mat"
        assert os.path.exists(temp), "data {} after downloading and extracting is not available at this {} address.".format(data_tag, temp)
        data = h5py.File(temp, 'r')
    x, y = generate_trainable_data(data, image_size, tag, main_dir)
    return x, y



# Written by Zhou
def preprocess_data(trainData, testData):
    '''
    Normalization of data. 
    argument: 
        trainData: the training data
        testData: the test data
    return: 
        trainData: the training data after preprocess
        testData: the test data after preprocess
    '''
    mean = np.mean(trainData, axis=0)
    trainData = trainData.astype(np.float32) - mean.astype(np.float32)
    testData = testData.astype(np.float32) - mean.astype(np.float32)

    trainData /= 255
    testData /= 255
    
    return trainData, testData


