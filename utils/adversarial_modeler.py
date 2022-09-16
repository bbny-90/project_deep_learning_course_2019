#!/usr/bin/env python3
# TF2
from tensorflow.keras import Input,Model
from tensorflow.keras.layers import BatchNormalization,Conv2D,MaxPool2D,Flatten,Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MSE
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random

class Modeler(object):
    def __init__(self, NN_input_shape=(64,64,3)):
        """
            NN_input_shape: Neural Network shape
            model : is a keras model (object)
        """
        self.image_size = NN_input_shape[0]
        self.num_channels = NN_input_shape[2]
        self.NN_input_shape = NN_input_shape
        self.model = None 

    def compile_model(self):
        """
            # The following model is basicly from
            # https://juejin.im/post/5c04e342f265da6165015391
        """
        inputs = Input(shape=self.NN_input_shape)
    
        model = BatchNormalization()(inputs)
        model = Conv2D(64, (7, 7), activation='relu', padding='same')(model)
        model = MaxPool2D(pool_size=(2, 2))(model)
    
        model = BatchNormalization()(model)
        model = Conv2D(128, (5, 5), activation='relu', padding='valid')(model)
        model = MaxPool2D(pool_size=(2, 2))(model)
    
        model = BatchNormalization()(model)
        model = Conv2D(256, (3, 3), activation='relu', padding='valid')(model)
        model = MaxPool2D(pool_size=(2, 2))(model)
        model = Dropout(0.5)(model)
    
        model = Flatten()(model)
        model = Dense(1024, activation='relu')(model)
        model = Dense(512, activation='relu')(model)
    
        x1 = Dense(11, activation='softmax')(model)
        x2 = Dense(11, activation='softmax')(model)
        x3 = Dense(11, activation='softmax')(model)
        x4 = Dense(11, activation='softmax')(model)
        x5 = Dense(11, activation='softmax')(model)
        x6 = Dense(11, activation='softmax')(model)
       
        x = [x1, x2, x3, x4, x5, x6]
        
        model = Model(inputs = inputs, outputs = x)
        
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model
    
        
    def fit_model(self, x, y):
        """
            this function trains the model for data pair (x,y)
        """
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=2)
        history = self.model.fit(x, y, validation_split=0.2, batch_size=128,
                                  epochs=20, callbacks=[early_stopping])
        return history
    def save_model(self, path="./models/baseline_no_extra.h5"):
        """
            this functions saves the model in the "path"
        """
        self.model.save(path)
        
    def load_model(self, path="./models/baseline_no_extra.h5"):
        """
            this functions loads the model from the "path"
        """
        self.model = load_model('./models/baseline_no_extra.h5')
        
    def get_loss_accuracy_model(self, x,y):
        """
            this functions gives the model performance for the data pair (x,y)
        """
        result = self.model.evaluate(x, y)
        return result
    
    def extract_y_model_for_one_sample(self, y_model, sample_id):
        """
            this functions gives back the model y label ("y_model") in the format
            of raw data, then we can compare more easily.
            Inputs:
            y_model: batch of all model labels
            sample_id: the data point id you are intrested to extrat
            Output:
            res: y_label for the point id "sample_id" in the format of raw data
        """
        res = []
        for i in range(6):
            res.append(y_model[i][sample_id,:].reshape(1,11))
        return res
    
    def creat_adversarial_x(self, x, y_model, epsilon=0.1):
        """
            This functions creats an adversial example for the (x,y_model) pair. the idea
            and how we can use Keras and TF2 comes from 
            https://medium.com/analytics-vidhya/implementing-adversarial-attacks-and-defenses-in-keras-tensorflow-2-0-cab6120c5715
            Inputs:
            x: model inout
            y_model: model_label
            epsilon: adversial hyperparametr which controls the level of randomness
            Outpu:
            x_adversarial: and adversial image with the similar format as "x"
        """
        size = self.image_size
        x_tmp = x.reshape(1,size, size, self.num_channels)
        x_tmp = tf.cast(x_tmp, tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x_tmp)
            prediction = self.model(x_tmp)
            loss = MSE(y_model, prediction)
        gradient = tape.gradient(loss, x_tmp)
        signed_grad = tf.sign(gradient).numpy()
        x_adversarial = x_tmp + signed_grad * epsilon
        return x_adversarial.numpy()
    
    def plot_image(self, image, num_ch, name2save=None):
        """
            this function plots an image.
            Inputs:
            image: image 
            num_ch: image number of channels
            name2save: the path if you want to save this image
        """
        try:
            img_rows, img_cols, _ = image.shape
        except:
            _, img_rows, img_cols, _ = image.shape
        if num_ch == 1:
            out = image.reshape(img_rows, img_cols)
            #plt.imshow(out)
            plt.imshow((out * 255).astype(np.uint8))
        else:
            out = image.reshape((img_rows, img_cols, num_ch))
            #plt.imshow(out)
            plt.imshow((out * 255).astype(np.uint8))
        if (name2save):
            plt.savefig('./results/'+ name2save)
        plt.show()            

            
    def generate_adversarials(self, batch_size, x, y_model, epsilon=0.1, data_size=10000):
        """
            This functions generate a batch of adversial examples. 
            Refer to "creat_adversarial_x" for more info.
        """
        img_size = self.image_size
        num_ch = self.num_channels
        x_temp = np.zeros(shape=(batch_size, img_size,img_size,num_ch),dtype=float)
        y_model_temp = []
        for i in range(6):
            y_model_temp.append(np.zeros(shape=(batch_size,11),dtype=float))
        
        for batch in range(batch_size):
            N = random.randint(0, data_size)
            x_ex = x[N]
            y_model_ex = self.extract_y_model_for_one_sample(y_model, N)
            x_adv_ex = self.creat_adversarial_x(x_ex, y_model_ex, epsilon)
            x_temp[batch] = x_adv_ex
            for i in range(6):
                y_model_temp[i][batch,:] = y_model_ex[i]
       
        return x_temp, y_model_temp
        
    