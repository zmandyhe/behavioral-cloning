# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## Overview
---
In this project, you will use what you've learned about deep neural networks and convolutional neural networks to clone driving behavior. You will train, validate and test a model using Keras. The model will output a steering angle to an autonomous vehicle using a simulator provided by Udacity where you can steer a car around a track for data collection. You'll use image data and steering angles to train a neural network and then use this model to drive the car autonomously around the track.

**The goals / steps of this project are the following:**
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

## Project Steps
### Data Collection
Since we are training the model to drive within the line, it is important to collect the training data of keeping my car in the center as much as possible. If we just teach our car to drive in the center of the track then it will never know what to do if it comes on the side path, so I alsoneed to collect recovery data for that as well. Particularly, Udacity encourages including "recovery" data while training. This means that data should be captured starting from the point of approaching the edge of the track (perhaps nearly missing a turn and almost driving off the track) and recording the process of steering the car back toward the center of the track to give the model a chance to learn recovery behavior. It's easy enough for experienced humans to drive the car reliably around the track, but if the model has never experienced being too close to the edge and then finds itself in just that situation it won't know how to react.  So overall, i split my data collection into three phases:
* My first step of data are using Udacity's sample IMG data and driving log. I trained a basic model for 40 epochs using 15,230 images for training and validations. 
* I then focus on collecting recovery data in several driving curves that are challenging for my first trained model. 
* My last step is to use the challenge track to collect training data to enhance my variaties of my training data to cover diversity of curves, road condition, and more challenging recovery runs.

Since we are teaching the model to learn from the data, I found the more diveristy of the data we collected, the more advanced model we can train. For each phase of data collection, I used the transfer learning to load my previous trained model to re-train on new data sets. This turns out a good data collection strategy for this project.

### Data Preprocessing 
**Correction Factor for Steering Angles from Left and Right Cameras**
Our image data have 3 images corresponding to one steering angle that is captured from the center position camera. Scine we also have its corresponding left image and right image, we can apply a correction factore to correspond to those two images for center image to correctly refelct the angel. Infact we need to introduce a correction factor that we will add in the images taken from left dashboard camera and a correction factor that we will subtract in the images taken from right dashboard camera in order to keep our vehicle in the center. The numbers comes from experiements, so according to our experience, I set up the  steering correction angle as 0.2. So in case of left images, steering angle = steering angle  from center image + 0.2, and in case of right images, steering angle= steering angle from center image- 0.2. These data are read in as training data.

**Setting Up Cropping Layer**
Keras provides the Cropping2D layer for image cropping within the model. This is relatively fast, because the model is parallelized on the GPU, so many images are cropped simultaneously.
Also, by adding the cropping layer, the model will automatically crop the input images when make predictions in drive.py.
```
from keras.models import Sequential, Model
from keras.layers import Cropping2D
model = Sequential()
model.add(Cropping2D(cropping=((50,20),(0,0),input_shape=(160,320,3))))
```

**Image Resizing**
The input image size for the Nvidia CNNC model is (66,200,3), I resized the image with the following code:
```
height = 66
width = 200
dim = (width, height)
res_img = cv2.resize(image,dim,interpolation=cv2.INTER_LINEAR)
```
**Image Flipping**
Another effective technique for helping with the left turn bias is to flip images and taking the opposite sign of the steering angles. Below is the code to implement it:
```
mg_flipped = cv2.flip(image, 1)
angle_flipped = -angle
```
### Using Keras Generator to Reduce Memory Exhausion
Generators is a great way to work with large amounts of data like this project.  Instead of storing the preprocessed data in memory all at once, using a generator can pull pieces of the data and process them on the fly only when they are needed, which is much more memory-efficient. Keras has a generator libray to implement for this project as follows:
```
def data_generator(img_paths, angles, batch_size):
    img_paths, angles = shuffle(img_paths, angles)
    X = []
    y = []
    while True:
        for i in range(batch_size):
            img = plt.imread(img_paths[i])
            img = resize_image(img)
            angle = angles[i]
            X.append(np.array(img))
            np.squeeze(X)
            y.append(np.float32(angle))
        yield(np.squeeze(np.array(X)),np.float32(y))
```
When use the generator from model.fit_generator(), it calls as follows:
```
batch_size = 64
train_generator = data_generator(X_train_img_paths,y_train,batch_size=batch_size)
valid_generator = data_generator(X_valid_img_paths,y_valid,batch_size=batch_size)
```

### Keras CNN Model Architecture
In this project, I followed Nvidia's [End to End Learning for Self-Driving Cars](  ) CNN architecutre. The network consists of 9 layers, including a normalization layer using Keras's [lambda layers](https://keras.io/layers/core/#lambda) to create arbitrary functions that operate on each image as it passes through each layer. The lambda layer will also ensure that the model will normalize the input images when making predictions in drive.py.
![CNN Architecture](https://github.com/zmandyhe/behavioral-cloning/blob/master/pic/nvidia-cnn.png)

```
def model(loss='mse', optimizer='adam'):
    model = Sequential()

    # set up lambda layer and normalize the input
    # model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(66,200,3)))
    model.add(Lambda(lambda x: x/127.5 - 1.0,input_shape=(66,200,3)))

    # Add three 5x5 convolution layers each with 2x2 stride
    # 1st: output 24@31*98
    model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2), padding = 'valid', activation='relu'))
    #2nd: output 36@14*47
    model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2),padding = 'valid', activation='relu'))
    # 3rd: output 48@5*22
    model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2),padding = 'valid', activation='relu'))

    #model.add(Dropout(0.50))

    # Add two 3x3 convolution layers with non-strided
    # 1st: 64@3*20
    model.add(Conv2D(64, kernel_size=(3, 3),padding = 'valid', activation='relu'))
    # 2nd: 64@1*18
    model.add(Conv2D(64, kernel_size=(3, 3), padding = 'valid', activation='relu'))

    # Add a flatten layer
    model.add(Flatten())

    # Add three fully connected layers,tanh activation
    model.add(Dense(100, activation = 'relu'))
    #model.add(Dropout(0.50))
    model.add(Dense(50, activation = 'relu'))
    #model.add(Dropout(0.50))
    model.add(Dense(10, activation = 'relu'))
    #model.add(Dropout(0.50))

    # Add a fully connected output layer for vehicle control
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer=optimizer)

    return model
```

### Model Training and Validation
To train the model, I run Keras's model.fit_generator() function with the setting of training and validation data read from the generator with the corresponding batch_size on each epoch. 
```
history = model.fit_generator(generator = train_generator,
                                steps_per_epoch = 5,
                                epochs = nb_epoch,
                                validation_steps=1,
                                validation_data = valid_generator,
                                verbose = 1,
                                callbacks=[checkpoint])
```

### Test and Record Automonous Driving in the Simulator
After the model is saved to "model.h5", I used the Udacity's emulator and drive.py to test on the autonomous driving model. This command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection. The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.
```
python3 drive.py model.h5 run1
```

The foloder of 'run1' save all the frames during the autonomous driving which looks like as follows. The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...

I then create a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.
```
python video.py run1 --fps 48
```
Will run the video at 48 FPS. The default FPS is 60.

### Results


### Reproduce This Project
If you don't have the lab environment, you can download and install it from:
* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)
* The simulator can be downloaded from GitHub [Udacity Simulator](https://github.com/udacity/self-driving-car-sim). You can use the "training mode" to collect training datasets directly from the sitmulator, which produces video frame images and driving_log.csv with all the 3 cameras view images and steering angle data.
* To train your model, run the "python3 behavioral-cloning.py"
* To test on autonomous mode, run "python 3 drive.py model.h5 run1"
* To save the autonomous run from "run1", run"python video.py run1 --fps 48"

## Reference Resources
Here is a list of resources that provide useful insights:
* 




