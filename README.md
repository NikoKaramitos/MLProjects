# MLProjects
# Convolutional Neural Network Project

## Prepared by: Niko Karamitos

## Description:
This project demonstrates a convolutional neural network (CNN) model for classifying images from the CIFAR10 dataset using PyTorch.

## Table of Contents:
Import Libraries
Loading necessary libraries such as torch, torchvision, and more.
Data Preparation
Image Transformations: Applying augmentations like Color Jittering, Gaussian Blur, etc.
Train and Test Set Preparation: Loading CIFAR10 dataset.
### Data Splitting: 
Creating a validation set from the training dataset.
### Dataloaders: 
Initializing dataloaders for training, validation, and test datasets.
### Displaying Images: 
Viewing a random batch of training images.
### Model Design
### Neural Network Structure: 
Defining the CNN architecture.
### Model Initialization: 
Creating an instance of the model.
### Loss and Optimizer: 
Setting up CrossEntropy loss and the Adam optimizer.

## Training and Validation
### Training Loop: 
Running the training loop for the model.
### Visualization: 
Plotting training and validation loss, and accuracy over epochs.
## Testing on New Data
### Loading Best Model: 
Loading the best model from the saved states.
### Testing: 
Taking a random batch of test set images, showing them, and computing model output for predictions.
## Usage:
The project begins by importing all the necessary libraries.
CIFAR10 dataset is loaded, transformed, and split into training, validation, and test sets.
A CNN model 'Niko' is defined, followed by model training over several epochs. Model states are saved every 5 epochs.
Finally, the best model state is loaded for testing and evaluating on a batch of test data.
