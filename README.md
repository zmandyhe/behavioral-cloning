# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## Overview
In this project, I will use deep neural networks and convolutional neural networks to clone driving behavior. I will train, validate and test a model using Keras and TensorFlow as backend. The model will output a steering angle to an autonomous vehicle using a simulator provided by Udacity. I'll use image data and steering angles to train a neural network and then use this model to drive the car autonomously around the track.

**The steps of this project are the following:**
* Use the simulator to collect data of good driving behavior in varied road conditions and driving conditions
* Design, train and validate a CNN model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

## Project Steps
### Data Collection
Since I am going to training the model to drive within the line, it is important to collect the training data of keeping my car in the center as much as possible. However, this type of data only works well for straight lane and works badly for curves and transactions among different road conditions. If we just teach our car to drive in the center of the track then it will never know what to do if it comes on the side path, so I also intentionly collected recovery data for left and right curved tracks for that purpose. Particularly, Udacity encourages including "recovery" data while training. This means that data should be captured starting from the point of approaching the edge of the track nearly missing a turn and almost driving off the track and recording the process of steering the car back toward the center of the track to give the model a chance to learn recovery behavior. It's easy enough for experienced humans to drive the car reliably around the track, but if the model has never experienced being too close to the edge and then finds itself in just that situation it won't know how to react.  But I found collecting the  recovery data is a little  tricky  and needs some trial-and-error, as the previous image data and angles data would impact where the car turns to for the next several timestamps, I found to collect data for curved lanes, it is best to start from a straight lane several time stamps before the curves happening to ensure the driving behavior during the tranction works efficiently.

 Overall, I split my data collection into three phases:
* My first step of data are using Udacity's sample IMG data and driving log. I trained a basic model to evaluate the model performance and where might be possible to improve.
* I then focus on collecting different sets of recovery data in several driving curves that are challenging for my first trained model.  I used the data to train and test the prediction repeatly.
* After I found a combination of dataset that works 98% correctly, I stopped the data collection (as more data are not always helpful) and used the twice of the dataset to retrain the final model.

### Data Preprocessing 
**Correction Factor for Steering Angles from Left and Right Cameras**
Our image data have 3 images corresponding to one steering angle that is captured from the center position camera. Scine we also have its corresponding left image and right image, we can apply a correction factor to correspond to those two images to correctly refelct the angel. Infact we need to introduce a correction factor that we will add in the images taken from left dashboard camera and a correction factor that we will subtract in the images taken from right dashboard camera in order to keep our vehicle in the center. The numbers comes from experiements, so according to our experience, I set up the  steering correction angle as 0.2. So in the case of left images, "steering angle = steering angle  from left image + 0.2", and in the case of right images, "steering angle= steering angle from right image- 0.2". These data are read in as training data.

**Image Cropping**
As each image data contains quite a portion that has no help to the lane conditions (e.g. trees in the far away), so I applied image cropping in mode.py and drive.py. Before other data pre-processing in both training and prediction, images are moved the a portion of top and a portion of bottom. The purpose is to remove the non-features to increase model performance.
```
def cropping_image(image): 
    x1, y1 = 0, 40
    x2, y2 = 320, 140
    roi = image[y1:y2,x1:x2]
    return roi
```

**Image Resizing**
The input image size for the Nvidia CNNC model is (66,200,3), I resized the image with the following code:
```
def resize_image(image):
    # setting dim of the resize
    height = 66
    width = 200
    dim = (width, height)
    res_img = cv2.resize(image,dim,interpolation=cv2.INTER_LINEAR)
    return res_img
```
**Image Flipping:**
Another effective technique for helping with the left turn bias and right turn bias is to flip images and taking the opposite sign of the steering angles. I implemented it in the Generator(), below is the code to implement it:
```
 if abs(angle)-0.2 > 0.1:
    img_flipped,angle_flipped = flipimg(img,angle)
    images.append(np.array(img_flipped))
    np.squeeze(images)
    steering_angles.append(np.float32(angle_flipped))
```
### Using Keras Generator to Efficiently Load the Data on the fly
Generators is a great way to work with large amounts of data like this project.  Instead of storing the preprocessed data in memory all at once, using a generator can pull pieces of the data and process them on the fly only when they are needed, which is much more memory-efficient. Keras has a generator libray to implement for this project as follows:
```
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
```
When use the generator from model.fit_generator(), it calls as follows:
```
batch_size = 768
train_generator = data_generator(X_train_img_paths,y_train,batch_size=batch_size)
valid_generator = data_generator(X_valid_img_paths,y_valid,batch_size=batch_size)
```

### Keras CNN Model Architecture
In this project, I followed Nvidia's paper of [End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316) CNN architecutre. The network consists of 9 layers, including a normalization layer using Keras's [lambda layers](https://keras.io/layers/core/#lambda) to create arbitrary functions that operate on each image as it passes through each layer. The lambda layer will also ensure that the model will normalize the input images when making predictions in drive.py. I used the RGB format for the input image with the size of (66,200,3).

**experiment with dropout and Kernel Regularizer**
Regularizers allow to apply penalties on layer parameters or layer activity during optimization. These penalties are incorporated in the loss function that the network optimizes. The penalties are applied on a per-layer basis in my network. By combining with one dropout in the first flat layer, the network greatly reduced t over-fitting.

**The Model Architecture and Definition**
![CNN Architecture](https://github.com/zmandyhe/behavioral-cloning/blob/master/pic/nvidia-cnn.png)
```
def model(loss='mse', optimizer='adam'):
    model = Sequential()
    
    # set up lambda layer and normalize the input
    model.add(Lambda(lambda x: x/127.5 - 1.0,input_shape=(66,200,3)))
    
    # Add three 5x5 convolution layers each with 2x2 stride
    # 1st: output 24@31*98
    model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2), padding = 'valid', activation='relu',  
						kernel_regularizer=regularizers.l2(0.001)))
    #2nd: output 36@14*47
    model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2),padding = 'valid', activation='relu',
						kernel_regularizer=regularizers.l2(0.001)))
    # 3rd: output 48@5*22
    model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2),padding = 'valid', activation='relu'			
										 kernel_regularizer=regularizers.l2(0.001)))

    # Add two 3x3 convolution layers with non-strided
    # 1st: 64@3*20
    model.add(Conv2D(64, kernel_size=(3, 3),padding = 'valid', activation='relu', 
						kernel_regularizer=regularizers.l2(0.001)))
    # 2nd: 64@1*18
    model.add(Conv2D(64, kernel_size=(3, 3), padding = 'valid', activation='relu', 
						kernel_regularizer=regularizers.l2(0.001)))

    # Add a flatten layer
    model.add(Flatten())

    # Add three fully connected layers,tanh activation
    model.add(Dense(100, activation = 'relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.50))
    model.add(Dense(50, activation = 'relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(10, activation = 'relu', kernel_regularizer=regularizers.l2(0.001)))

    # Add a fully connected output layer for vehicle control
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer=optimizer)

    return model
```

### Model Training and Validation
To train the model, I run Keras's model.fit_generator() function with the setting of training and validation data read from the generator with the corresponding batch_size on each epoch. 
```
nb_epoch = 30
history = model.fit_generator(generator = train_generator,
                            steps_per_epoch = 10,
                            epochs = nb_epoch,
                            validation_steps=2,
                            validation_data = valid_generator,
                            verbose = 2)
```

I used the Adam optimizer in the model, and adjust the learning from its default rate of 0.001 to 0.0001.
```
optimizer = Adam(lr=0.0001)
model = model(loss='mse', optimizer=optimizer)
```

### Test and Record Automonous Driving in the Simulator
After the model is saved to "model.h5", I used the Udacity's emulator and the slightly customized drive.py (added image preprocessing) to test the autonomous driving mode in the simulator. This command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection. The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.
```
python3 drive.py model.h5 run1
```

The foloder of 'run1' save all the frames during the autonomous driving which looks like as follows. The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving called run1.mp4.
![run1 image](https://github.com/zmandyhe/behavioral-cloning/blob/master/pic/run1-img-list.png)

I then create a video based on images found in the above `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.
```
python video.py run1 --fps 48
```
Will run the video at 48 FPS. The default FPS is 60.

### Results
I used the final dataset which combined Udacity's and my own, and included a total of 92,085 data points. I finally chose to implement only one 50% dropout after the first fully connected layer, so the final data samples passed to the final layer for each epoch are 92,085. These data are remained after distribution flattening, and this set was further split into a training set of 73,668 (80%) data points and a validation set of 18,417 (20%) data points. 

The model contains 252,219 parameters in total, which are all trainable params (252,219), 0 non-trainable params. The kernel regularizer for the model's each layer and the one dropout works fairly good to control overfitting. The final training loss is 0.2176, validation loss is 0.2326. These were not the lowest loss values in my experiments, but the model with these weights performed even better than those model weights with lower loss value on the autonomous driving in the sitmulator. 

### Conclusion and Discussion
I am excited to see the vehicle drives autonomously and smoothly through my trained model. Through the many trial-and-error process involved many hours, I was able to develop a deeper understanding about the relationship between the training data as human's driving behaviors and the vehicle autonomous cloning behaviors, which is essential to this project. In summary, I have the following insights from completing this project:
* When we collect the human driving behavior data, we shall show a strong intension of our behaviors. Which means when we drive on a sharp left curve, our intention is to steer our wheel along the left curve to stay left as closely as possible so that our car would not be thrown out of the road to the right side. This reflects to the data that there shall be many steering angles in the dataset.
* In the training model on the emulator, speeding up so that we will collect as many as discovery data as we can.
* During data collection, the transaction period of time from a straight line road condition to sharpe curves is essential to show our driving behavioral intention. For example, when we see a left curve ahead from a straight lane line, we shall adjust our training vehicle gradually to the left to prepare for the left curve. I can see the model improvement of having this type of data in the autonomous mode during the model improvement process, then we will know what other data (intentions) we will need to collect to feed in the model to improve it.
* Collecting good data is key in this project, as we have the NVIDA's CNN model or with models already available to start with. But fine tuning and testing with varied model architecutres to find the most effective model architecture is interesting to me in the next phase.
* Very small validation loss value does not correlate with a good model in this project. But through techniques of avoiding overfitting and ensuring a good training data, when we see the model is converging, we have a good sense to guess that the model would work.

### Reproduce This Project
Follow below process to reproduce the training and prediction process:
* If you don't have the lab environment, you can download and install it from [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)
* The simulator can be downloaded from GitHub [Udacity Simulator](https://github.com/udacity/self-driving-car-sim). You can use the "training mode" to collect training datasets directly from the sitmulator, which produces video frame images and driving_log.csv with all the 3 cameras view images and steering angle data.
* To train your model, run the "python3 model.py"
* To test on autonomous mode, run "python 3 drive.py model.h5 run1"
* To save the autonomous run from "run1" to video, run"python video.py run1 --fps 48"





