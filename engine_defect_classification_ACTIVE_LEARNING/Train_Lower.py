import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
import time
import matplotlib.pylab as plt
import PIL.Image as Image
import numpy as np
import pandas as pd
import tensorflow.keras as keras
import os
import random
import pickle
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import  Dropout, Input
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import itertools
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras import optimizers
from tensorflow.python.client import device_lib
from tensorflow.keras.layers import Input, UpSampling2D, Flatten, BatchNormalization, Dense, Dropout, GlobalAveragePooling2D
from keras.backend.tensorflow_backend import set_session
from augment_misclassfied_upper import *
from crop_images import *
import argparse
from keras import backend as K
import glob

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

tf.debugging.set_log_device_placement(True)
from tensorflow.python.client import device_lib


import time
from sklearn.metrics import classification_report,f1_score,precision_score,recall_score,confusion_matrix,accuracy_score
import itertools
from matplotlib import pyplot as plt
# %matplotlib inline
    
## GET THE AVAILABILITY OF GPU
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def Delete_random_files(path,x):
    list_of_imgs = os.listdir(path)
    imgs_to_be_deleted = np.random.choice(list_of_imgs,x,replace=False)
    for img in imgs_to_be_deleted:
        if(img== ".ipynb_checkpoints"):
            continue
        if(img== "kpoints"):
            continue
        os.remove(os.path.join(path,img))
    

## function to create model
def create_model():
    with tf.device('/device:GPU:0'):
        resnet_model = tf.keras.applications.resnet.ResNet101(weights='imagenet', include_top=False, input_tensor=Input(shape=(96,96,3)))
        for layer in resnet_model.layers:
            if isinstance(layer, BatchNormalization):
                layer.trainable = True
            else:
                layer.trainable = False

        model = Sequential()
        model.add(resnet_model)
        model.add(GlobalAveragePooling2D())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(.5))
        model.add(BatchNormalization())
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=["accuracy"])
        return model

## Function to prepare training and testing dataset
def create_datagen(train_path,test_path):
    batch_size = 32

    train_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
            train_path,  # this is the target directory
            target_size=(96, 96),
            classes =  ['0', '1'],
            batch_size=batch_size,
            shuffle=True,
            class_mode='categorical')
    batch_size=32
    
    predict_datagen = ImageDataGenerator( rescale = 1./255. )
    predict_generator =  predict_datagen.flow_from_directory(
                        test_path,
                        batch_size=batch_size,
                        classes =  ['0', '1'],
                        class_mode='categorical',
                        shuffle=False,
                        target_size=(96, 96))
    return train_generator,predict_generator

## Function to train model
def train_model(data_path,model_ckpt_path):
    train_path = data_path+'/train_new_augmented/'
    test_path = data_path+'/validation/'
    batch_size=32
    train_generator,validation_generator=create_datagen(train_path,test_path)
    
    total_train_images=0
    for i in range(2):
        total_train_images=total_train_images+len(glob.glob(os.path.join(train_path+str(i), '*')))
    total_validation_images=0
    for i in range(2):
        total_validation_images=total_validation_images+len(glob.glob(os.path.join(test_path+str(i), '*')))
    print("Total_train_images",total_train_images)
    print("Total_validation_images",total_validation_images)
    
    with tf.device('/device:GPU:0'):
        callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=model_ckpt_path+'/saved-model-augmented-gpu-{epoch:02d}.h5',monitor='val_loss',verbose=1, save_best_only=False, save_weights_only=False, mode='auto',period=1)
#                                                         ,tf.keras.callbacks.EarlyStopping(patience=2)
]
        
        model=create_model()
        print("Training is started..")
        history=model.fit_generator(
                train_generator,
                steps_per_epoch=total_train_images// batch_size,
                epochs=30,
                validation_data=validation_generator,
                validation_steps=total_validation_images// batch_size,callbacks=callbacks)
        print("Training Done!!")
        
## prediction on test dataset       
def prediction(test_data_generator, model_path):
    load_model= tf.keras.models.load_model(model_path)
    
    start_time=time.time()
    y_pred= load_model.predict_generator(test_data_generator,verbose=1)
    end_time=time.time()
    print(end_time-start_time)
    
    labels = ["non-defective","defective"]
    
    true_values = test_data_generator.classes
    predicted_values = np.argmax(y_pred, axis=1)
    
    print(classification_report(true_values, predicted_values, target_names=labels))
    
    test_accuracy = np.sum(predicted_values == true_values) / len(true_values)
    print ("Test Accuracy:", test_accuracy)
    
    cm_test = confusion_matrix(true_values, predicted_values)
    print("Confusion Matrix")
    print(cm_test)
    return test_accuracy
    
## Compare the results for retrained model on new dataset and original trained model on original data   
def compare_model(old_model_path, Retrained_model_path, test_data_generator):
    print("Prediction Starting using old model-----")
    acc1 = prediction(test_data_generator,old_model_path)
    print("Prediction Starting using Retrained model-----")
    acc2 = prediction(test_data_generator,Retrained_model_path)
    if(acc1>acc2):
        return 1
    if(acc2>acc1):
        return 2
    if(acc2==acc1):
        print("Equal")
        return 2
    else:
        return -1
    
    
    
if __name__ == "__main__":
    # Setup the Argument parsing
    parser = argparse.ArgumentParser(prog='train_resnet101', description='Script which trains a ResNet101 Upper Classification model')

    parser.add_argument('--old_data_path', dest='old_data_filepath', type=str, help='Train data to use for (Required)', required=True)
    parser.add_argument('--new_data_path', dest='new_data_filepath', type=str, help='Test data to use for testing (Required)', required=True)
    parser.add_argument('--output_dir', dest='output_folder', type=str, help='Folder where checkpoints will be saved(Required)', required=True)

    parser.add_argument('--old_model_path', dest='old_model_filepath', type=str, help='Old model weights path of 100th epoch', required=True)
    parser.add_argument('--final_model_path', dest='final_model_filepath', type=str, help='Final model weights folder', required=True)
    

    print(parser.parse_args())
    args = parser.parse_args()
    model_ckpt_path= args.output_folder
    old_data_path = args.old_data_filepath
    new_data_path = args.new_data_filepath
    old_model_path=args.old_model_filepath
    final_model_path=args.final_model_filepath
    
    
    ## get the images crooped into lower and upper datasets..
    defective_path = new_data_path + "/defective"
    non_defective_path = new_data_path + "/nondefective"

    ## Output folders
    new_data_path_lower = new_data_path + "/Lower"
#     new_data_path_upper = new_data_path + "/Upper"
    
    ## Get Lower and Upper Data separated..
    get_data_lower(defective_path, non_defective_path, new_data_path_lower)
    
    # get the new images, augment them and merge to original images
    x = run_augment(new_data_path_lower)
    if(x != 0):
        Delete_random_files(old_data_path+"/train/0",x)
        
    Copy(new_data_path_lower+'/train_new_augmented/1',old_data_path+'/train_new_augmented/1')
    Copy(new_data_path_lower+'/train_new_augmented/0',old_data_path+'/train_new_augmented/0')
    Copy(new_data_path_lower+'/test/1',old_data_path+'/validation/1')
    Copy(new_data_path_lower+'/test/0',old_data_path+'/validation/0')
    
    final_data_path = old_data_path
    
    ## Perform re-training
#     train_model(final_data_path,model_ckpt_path)
    
    ## Prepare data for prediction
    train_path=final_data_path+'/train_new_augmented/'
    test_path=final_data_path+'/validation/'
    train_generator,validation_generator=create_datagen(train_path,test_path)
    
    ## get models original and retrained model for prediction, hence comparision
    model_path_1 = old_model_path
    retrain_model_path = model_ckpt_path+'/saved-model-augmented-gpu-30.h5'
    m = compare_model(model_path_1,retrain_model_path, validation_generator)
    print(m)
    if(m==1):
        model_path = model_path_1
    if(m==2):
        model_path = retrain_model_path
    print(model_path)

        