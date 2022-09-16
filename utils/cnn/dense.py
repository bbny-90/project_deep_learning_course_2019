from tensorflow.keras import Input,Model
from tensorflow.keras.layers import BatchNormalization,Conv2D,MaxPool2D,Flatten,Dense,Dropout

def dense_model(input_shape=(64,64,3)):
    inputs = Input(shape=input_shape)
    
    model = BatchNormalization()(inputs)
    model = Conv2D(64, (7, 7), activation='relu', padding='same')(model)
    model = MaxPool2D(pool_size=(2, 2))(model)
    
    model = BatchNormalization()(model)
    model = Conv2D(128, (5, 5), activation='relu', padding='same')(model)
    model = MaxPool2D(pool_size=(2, 2), strides=1, padding='same')(model)
    
    model = Conv2D(196, (5, 5), activation='relu', padding='valid')(model)
    model = MaxPool2D(pool_size=(2, 2))(model)
    
    model = BatchNormalization()(model)
    model = Conv2D(196, (3, 3), activation='relu', padding='same')(model)
    model = MaxPool2D(pool_size=(2, 2), strides=1, padding='same')(model)
    
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
    
    return model
