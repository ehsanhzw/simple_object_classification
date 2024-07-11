## Neural Network Models:
from keras import models, layers, regularizers
def cnn_model(input_shape=(32,32,3)):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3,3), activation='relu',input_shape=input_shape, padding='same'))
    model.add(layers.Dropout(0.1))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(layers.Dropout(0.1))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(layers.Dropout(0.1))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(layers.Dropout(0.1))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(10, activation='softmax'))
    return model

def unet_model(input_shape=(32,32,3)):
    inputs = layers.Input(input_shape)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = layers.Dropout(0.05)(c1)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    c2 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = layers.Dropout(0.1)(c2)
    c2 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    c3 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = layers.Dropout(0.1)(c3)
    c3 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    c4 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = layers.Dropout(0.1)(c4)
    c4 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)
    c5 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = layers.Dropout(0.2)(c5)
    c5 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    flat = layers.Flatten()(c5)
    fc1 = layers.Dense(256, activation='relu')(flat)
    d1 = layers.Dropout(0.2)(fc1)
    fc2 = layers.Dense(64, activation='relu')(d1)
    d2 = layers.Dropout(0.2)(fc2)
    outputs = layers.Dense(10, activation='softmax')(d2)
    model = models.Model(inputs,outputs)
    return model

def quickcnn_model(input_shape=(32,32,3)):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=input_shape))
    model.add(layers.Dropout(0.15))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(layers.Dropout(0.15))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.15))
    model.add(layers.Dense(10, activation='softmax'))
    return model

def param_model(input_shape=(32,32,3)):
    # https://github.com/Param-GG/Animal-Classification-CNN
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=input_shape))
    model.add(layers.Dropout(0.16))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(layers.Dropout(0.16))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(layers.Dropout(0.16))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(layers.Dropout(0.16))
    model.add(layers.Dense(10, activation='sigmoid'))
    return model

def slowunet_model(input_shape=(32,32,3)):
    # extremely slow and lots of convolutions
    inputs = layers.Input(input_shape)
    c1 = layers.Conv2D(64, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    c1 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(c1)
    p1 = layers.MaxPooling2D(pool_size=(2, 2))(c1)
    c2 = layers.Conv2D(128, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(p1)
    c2 = layers.Conv2D(128, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(c2)
    p2 = layers.MaxPooling2D(pool_size=(2, 2))(c2)
    c3 = layers.Conv2D(256, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(p2)
    c3 = layers.Conv2D(256, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(c3)
    p3 = layers.MaxPooling2D(pool_size=(2, 2))(c3)
    c4 = layers.Conv2D(512, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(p3)
    c4 = layers.Conv2D(512, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(c4)
    d4 = layers.Dropout(0.5)(c4)
    p4 = layers.MaxPooling2D(pool_size=(2, 2))(d4)
    c5 = layers.Conv2D(1024, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(p4)
    c5 = layers.Conv2D(1024, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(c5)
    d5 = layers.Dropout(0.5)(c5)
    flat = layers.Flatten()(d4)
    fc1 = layers.Dense(1024, activation='relu')(flat)
    d1 = layers.Dropout(0.2)(fc1)
    fc2 = layers.Dense(128, activation='relu')(d1)
    d2 = layers.Dropout(0.2)(fc2)
    outputs = layers.Dense(10, activation='softmax')(d2)
    model = models.Model(inputs, outputs)
    return model

def test_model(input_shape):
    inputs = layers.Input(shape=input_shape)

    ## TEST 16
    model = models.Sequential()
    model.add(layers.Input(input_shape))
    model.add(layers.Conv2D(35, (3,3), activation='relu', padding='same',
                            kernel_regularizer=regularizers.l1_l2(0.000,0.001)))
    model.add(layers.Dropout(0.2))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same',
                            kernel_regularizer=regularizers.l1_l2(0.000,0.001)))
    model.add(layers.Dropout(0.2))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Conv2D(128, (3,3), activation='relu', padding='same',
                            kernel_regularizer=regularizers.l1_l2(0.000,0.001)))
    model.add(layers.Dropout(0.2))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Conv2D(256, (3,3), activation='relu', padding='same',
                            kernel_regularizer=regularizers.l1_l2(0.000,0.001)))
    model.add(layers.Dropout(0.2))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l1_l2(0.000,0.002)))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l1_l2(0.000,0.002)))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(10, activation='softmax'))

    return model