from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import math
import numpy as np      
import cv2                 
import matplotlib.pyplot as plt
from os import getcwd
import csv
import tensorflow as tf
from scipy import ndimage
from keras.models import Sequential, Model
from keras.layers import Lambda, Cropping2D
from keras import regularizers

# Read in images paths for center,left,right cameras in each row
def read_in_image_path_with_space(csv_folder_path, img_path_foldername):
    csv_file = csv_folder_path + "driving_log.csv"
    img_paths,angles = ([],[])
    with open(csv_file,'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            # get center images and angles
            img_paths.append(img_path_foldername + row[0].strip())
            angles.append(row[3])
            # get left images and angles
            img_paths.append(img_path_foldername + row[1].strip())
            angles.append(float(row[3])+0.2)
            # get right images and angles
            img_paths.append(img_path_foldername + row[2].strip())
            angles.append(float(row[3])-0.2)
    return np.array(img_paths), np.array(angles)

def read_in_image_path_with_string(csv_folder_path, img_path_foldername):
    csv_file = csv_folder_path + "driving_log.csv"
    img_paths,angles = ([],[])
    with open(csv_file,'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            # get center images and angles
            path= (img_path_foldername + row[0].strip()).split('/')[-1]
            img_paths.append(img_path_foldername + path)
            angles.append(row[3])
            # get left images and angles
            path = (img_path_foldername + row[1].strip()).split('/')[-1]
            img_paths.append(img_path_foldername + path)
            angles.append(float(row[3])+0.2)
            # get right images and angles
            path = (img_path_foldername + row[2].strip()).split('/')[-1]
            img_paths.append(img_path_foldername + path)
            angles.append(float(row[3])-0.2)
    return np.array(img_paths), np.array(angles)

img_paths_1, angles_1 = read_in_image_path_with_space(csv_folder_path = "/opt/carnd_p3/data/", img_path_foldername = "/opt/carnd_p3/data/")
img_paths_3, angles_3 = read_in_image_path_with_string(csv_folder_path = "own-data-2/", img_path_foldername = "own-data-2/IMG/")
img_paths_4, angles_4 = read_in_image_path_with_string(csv_folder_path = "own-data-3/", img_path_foldername = "own-data-3/IMG/")
img_paths_5, angles_5 = read_in_image_path_with_string(csv_folder_path = "own-data-4/", img_path_foldername = "own-data-4/IMG/")
img_paths_6, angles_6 = read_in_image_path_with_string(csv_folder_path = "own-data-5/", img_path_foldername = "own-data-5/IMG/")
img_paths_7, angles_7 = read_in_image_path_with_string(csv_folder_path = "own-data-6/", img_path_foldername = "own-data-6/IMG/")
img_paths_8, angles_8 = read_in_image_path_with_string(csv_folder_path = "own-data-8/", img_path_foldername = "own-data-8/IMG/")
img_paths_9, angles_9 = read_in_image_path_with_string(csv_folder_path = "own-data-9/", img_path_foldername = "own-data-9/IMG/")
img_paths_21 = np.concatenate((img_paths_1, img_paths_3,  img_paths_4, img_paths_5, img_paths_6,img_paths_7, img_paths_8, img_paths_9),axis = 0)
img_paths = np.concatenate((img_paths_21, img_paths_21))
angles_21 = np.concatenate((angles_1,angles_3, angles_4, angles_5, angles_6,angles_7, angles_8,angles_9), axis = 0)
angles = np.concatenate((angles_21, angles_21))
img_paths = np.array(img_paths)
angels = np.float32(angles)

# read in sample image and its shape
def read_in_image(path):
    image = plt.imread(path)
#         image = ndimage.imread(img_paths[i])
    return image

# Crop the top and bottom of the image
def cropping_image(image): 
    x1, y1 = 0, 40
    x2, y2 = 320, 140
    roi = image[y1:y2,x1:x2]
    return roi
    
# resize to 66*200*3 to fit Nvidia input image size requirement
def resize_image(image):
    # setting dim of the resize
    height = 66
    width = 200
    dim = (width, height)
    res_img = cv2.resize(image,dim,interpolation=cv2.INTER_LINEAR)
    return res_img

# flipping image and take the opposite sign of the measurement
def flipimg(image,angle):
    img_flipped = cv2.flip(image, 1)
#     img_flipped = np.fliplr(image)
    angle_flipped = -angle
    return img_flipped,angle_flipped

# split data to training and validation sets
# shuffle image paths and angles for data generator
X_train_img_paths,X_valid_img_paths,y_train,y_valid = train_test_split(img_paths,angles,test_size=0.2,random_state=42)

# use generator to read in data
def generator(img_paths, angles, batch_size=128):
    num_samples = len(img_paths)
    while 1: # Loop forever so the generator never terminates
        shuffle(img_paths)
        for offset in range(0, num_samples, batch_size):
            batch_samples = img_paths[offset:offset+batch_size]

            images = []
            steering_angles = []
            for i, batch_sample in enumerate(batch_samples):
                img = plt.imread(img_paths[i])
                img = cropping_image(img)
                img = resize_image(img)
                angle = np.float32(angles[i])
                if abs(angle)-0.2 > 0.1:
                    img_flipped,angle_flipped = flipimg(img,angle)
                    images.append(np.array(img_flipped))
                    np.squeeze(images)
                    steering_angles.append(np.float32(angle_flipped))
                else:
                    images.append(np.array(img))
                    np.squeeze(images)
                    steering_angles.append(np.float32(angle))
            # trim image to only see section with road
            X = np.array(images)
            y = np.array(steering_angles)
            yield shuffle(X, y)

# get generators
# compile and train the model using the generator function
batch_size = 768
train_generator = generator(X_train_img_paths, y_train, batch_size=batch_size)
valid_generator = generator(X_valid_img_paths, y_valid, batch_size=batch_size)

# define CNN model architecture using Nvidia model
#In this project, a lambda layer is a convenient way to
# parallelize image normalization. The lambda layer will also ensure
# that the model will normalize input images when making predictions in drive.py

def model(loss='mse', optimizer='adam'):
    model = Sequential()
    
    # set up lambda layer and normalize the input
    # model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(66,200,3)))
    model.add(Lambda(lambda x: x/127.5 - 1.0,input_shape=(66,200,3)))
    
    # Add three 5x5 convolution layers each with 2x2 stride
    # 1st: output 24@31*98
    model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2), padding = 'valid', activation='relu',  kernel_regularizer=regularizers.l2(0.001)))
    #2nd: output 36@14*47
    model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2),padding = 'valid', activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    # 3rd: output 48@5*22
    model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2),padding = 'valid', activation='relu', kernel_regularizer=regularizers.l2(0.001)))

    #model.add(Dropout(0.50))

    # Add two 3x3 convolution layers with non-strided
    # 1st: 64@3*20
    model.add(Conv2D(64, kernel_size=(3, 3),padding = 'valid', activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    # 2nd: 64@1*18
    model.add(Conv2D(64, kernel_size=(3, 3), padding = 'valid', activation='relu', kernel_regularizer=regularizers.l2(0.001)))

    # Add a flatten layer
    model.add(Flatten())
#     model.add(Dropout(0.50))

    # Add three fully connected layers,relu activation
    model.add(Dense(100, activation = 'relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.50))
    model.add(Dense(50, activation = 'relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(10, activation = 'relu', kernel_regularizer=regularizers.l2(0.001)))

    # Add a fully connected output layer for vehicle control
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer=optimizer)

    return model

# main pipeline
# checkpoint = ModelCheckpoint('model{epoch:02d}.h5')

nb_epoch = 30
#lr=0.0005
optimizer = Adam(lr=0.0001)
model = model(loss='mse', optimizer=optimizer)

# Compile and train the model
history = model.fit_generator(generator = train_generator,
                            steps_per_epoch = 10,
                            epochs = nb_epoch,
                            validation_steps=2,
                            validation_data = valid_generator,
                            verbose = 2)
# print model summary
print(model.summary())

# save model data
model.save('model.h5')
model_json = model.to_json()
with open ('model_json','w') as f:
    f.write(model_json)
