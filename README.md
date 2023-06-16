# This code was created as the final project of the Artificial Neural Network course.

# Image Classification using Convolutional Neural Networks

This code demonstrates how to train a Convolutional Neural Network (CNN) for image classification using the CIFAR-10 dataset. It also includes an example of classifying custom images using the trained model.

## Prerequisites

Make sure you have the following libraries installed:

- matplotlib
- tensorflow
- tensorflow_datasets
- numpy
- opencv-python (cv2)
- pandas

You can install them using pip:


## Dataset

The CIFAR-10 dataset is used for training and testing the CNN. The dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. In this code, we filter the dataset to include only three classes: automobiles, cats, and dogs.

## Training the Model

1. The CIFAR-10 dataset is loaded using the `cifar10.load_data()` function. The dataset is then filtered to include only the specified classes using the `load_cifar10_classes()` function.

2. The images are preprocessed by dividing the pixel values by 255 to scale them between 0 and 1.

3. The labels are converted to one-hot encoding using `tf.keras.utils.to_categorical()`.

4. A CNN model is defined using the Sequential API from Keras. The model consists of two convolutional layers, two max-pooling layers, a flatten layer, and two fully connected layers with ReLU and softmax activations.

5. The model is compiled with the Adam optimizer, categorical cross-entropy loss function, and accuracy metric.

6. The model is trained using the `fit()` function with the training data and validation data. The training is performed for a specified number of epochs and batch size.

7. The trained model is saved to a file named `trained_model.h5`.

## Classifying Custom Images

1. The folder path containing the custom images is specified.

2. The images are read and preprocessed using the `read_images_from_folder()` function.

3. The trained model is loaded from the `trained_model.h5` file.

4. The images are classified using the trained model. The predicted class labels are stored in the `predicted_labels` list.

5. The images and their predicted class labels are displayed using matplotlib.

## Monitoring the Training Progress

The training and validation loss, as well as the training and validation accuracy, are plotted using matplotlib. The loss plot shows the decrease in loss over epochs, while the accuracy plot shows the increase in accuracy.

