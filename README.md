# Gender_Prediction_Using_Facial_data
Dataset : https://www.kaggle.com/datasets/ashwingupta3012/male-and-female-faces-dataset

# Cleaning dataset

## Libraries Used
- os
- re
- random
- numpy
- tqdm
- matplotlib
- cv2 (OpenCV)
- shutil

This project utilizes various Python libraries for tasks such as data manipulation, visualization, and image processing.

## Installation
You can install the required libraries using pip:
```bash
pip install numpy tqdm matplotlib opencv-python
```
## Face Detection and Cropping 
Face Detetction is done with the prebuild Model  of "haarcascade_frontalface_default"

Link : https://github.com/Sh-bharat/Face_Detection_using_Pretrained_Model
## Handling Exceptions 
Some Exceptions which are by mistake detected as face data by "haarcascade_frontalface_default" were detected and removed. as they will Lower the Model acccuracy.

# Creating Model

describing the usage of `tf.keras.preprocessing.image.ImageDataGenerator` for image data generation and augmentation:
## Dependencies
- TensorFlow
- NumPy
- Matplotlib
- OpenCV
- tqdm
```markdown
# Image Data Generation and Augmentation using ImageDataGenerator

## Overview
This Section demonstrates the use of TensorFlow's `tf.keras.preprocessing.image.ImageDataGenerator` for generating and augmenting image data. It's commonly used in deep learning projects.

## Introduction
ImageDataGenerator provides a flexible way to load images from disk, perform real-time data augmentation, and generate batches of images and labels for training deep learning models.

## Setup
Ensure you have TensorFlow installed:
```bash
pip install tensorflow
```

## Usage
1. Define an instance of ImageDataGenerator with desired augmentation parameters:
    ```python
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1/255,
        rotation_range=4,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.15
    )
    ```

2. Generate data for training and validation using `flow_from_directory`:
    ```python
    train_ds = image_generator.flow_from_directory(
        directory=os.getcwd() + str("/Dataset"),
        subset='training'
    )

    test_ds = image_generator.flow_from_directory(
        directory=os.getcwd() + str("/Dataset"),
        subset='validation'
    )
    ```

3. Train your deep learning model using the generated data:
    ```python
    model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=10
    )
    ```

## Parameters
- `rescale`: Rescales pixel values to the range [0, 1].
- `rotation_range`: Range for random rotations applied to images.
- `width_shift_range` and `height_shift_range`: Ranges for random horizontal and vertical shifts.
- `shear_range`: Range for random shearing transformations.
- `horizontal_flip`: Boolean for random horizontal flipping.
- `fill_mode`: Strategy for filling newly created pixels.
- `validation_split`: Fraction of images to reserve for validation.



```markdown

## Model Summary
The CNN model architecture is as follows:

```plaintext
Model: "sequential_9"
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
conv2d_25 (Conv2D)          (None, 254, 254, 32)      896       
                                                                 
max_pooling2d_24 (MaxPooli  (None, 127, 127, 32)      0         
ng2D)                                                           
                                                                 
conv2d_26 (Conv2D)          (None, 125, 125, 64)      18496     
                                                                 
max_pooling2d_25 (MaxPooli  (None, 62, 62, 64)        0         
ng2D)                                                           
                                                                 
conv2d_27 (Conv2D)          (None, 60, 60, 128)       73856     
                                                                 
max_pooling2d_26 (MaxPooli  (None, 30, 30, 128)       0         
ng2D)                                                           
                                                                 
flatten_8 (Flatten)         (None, 115200)            0         
                                                                 
dense_24 (Dense)            (None, 64)                7372864   
                                                                 
dense_25 (Dense)            (None, 32)                2080      
                                                                 
dense_26 (Dense)            (None, 2)                 66        
                                                                 
=================================================================
Total params: 7468258 (28.49 MB)
Trainable params: 7468258 (28.49 MB)
Non-trainable params: 0 (0.00 Byte)
```

## Model Performance
The trained model achieves an accuracy of 95.81% on the validation set, indicating its effectiveness in predicting gender from facial images.
The Model is saved as `Gender_detection.keras`
