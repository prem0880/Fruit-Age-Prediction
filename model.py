
import sys
import os
from keras.layers import *
from keras.optimizers import *
from keras.applications import *
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as k
from keras.models import load_model



nb_classes = 5 
based_model_last_block_layer_number = 126 
img_width, img_height = 100, 100 
batch_size = 32  
nb_epoch = 50  
learn_rate = 1e-4  
momentum = .9  
transformation_ratio = .05 

base_model = Xception(input_shape=(img_width, img_height, 3), weights='imagenet', include_top=False)
     
train_data_dir = '/home/danush/OwnDataset/train'
validation_data_dir = '/home/danush/OwnDataset/test' 

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(nb_classes, activation='softmax')(x)

model = Model(base_model.input, predictions)

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=transformation_ratio,
                                   shear_range=transformation_ratio,
                                   zoom_range=transformation_ratio,
                                   cval=transformation_ratio,
                                   horizontal_flip=True,
                                   vertical_flip=True)

validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size=(img_width, img_height),
                                                    batch_size=batch_size,
                                                    class_mode='categorical')
    
validation_generator = validation_datagen.flow_from_directory(validation_data_dir,
                                                              target_size=(img_width, img_height),
                                                              batch_size=batch_size,
                                                              class_mode='categorical')

model.compile(optimizer='nadam',
              loss='categorical_crossentropy', 
              metrics=['accuracy'])


model.fit_generator(train_generator,
                    steps_per_epoch=2640/32,
                    epochs=3,
                    validation_data=validation_generator,
                    validation_steps=886/32,
                    )


for layer in model.layers[:-20]:
    layer.trainable = False
for layer in model.layers[-20:]:
    layer.trainable = True

model.compile(optimizer='nadam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit_generator(train_generator,
                    steps_per_epoch=2640/32,
                    epochs=10,
                    validation_data=validation_generator,
                    validation_steps=886/32,
                    )


model.save('/home/danush/my_model.h5')

