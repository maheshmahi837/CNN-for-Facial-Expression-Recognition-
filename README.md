# Facial Expression Recognition using CNN and Adversarial Attacks

## Overview

This repository contains the implementation of a custom Convolutional Neural Network (CNN) for Facial Expression Recognition using the FER-2013 dataset. The project was developed as part of the **AI61002: Deep Learning Foundations and Applications** course assignment.

The project focuses on designing a fully convolutional residual CNN architecture for emotion classification, visualizing learned features using Grad-CAM, and evaluating model robustness using targeted FGSM adversarial attacks.

---

## Project Objectives

* Load and preprocess the FER-2013 facial expression dataset
* Build a custom Fully Convolutional CNN architecture
* Implement residual connections between convolutional layers
* Train the network using Adam optimizer
* Evaluate the model using classification metrics
* Generate Grad-CAM visualizations for model interpretability
* Implement targeted FGSM adversarial attacks
* Analyze model robustness against adversarial perturbations

---

## Technologies Used

* Python
* PyTorch
* Torchvision
* NumPy
* Matplotlib
* Scikit-learn
* OpenCV
* Jupyter Notebook

---

## Dataset

The project uses the **FER-2013** dataset containing grayscale facial images classified into the following emotion categories:

* Angry
* Disgust
* Fear
* Happy
* Sad
* Surprise
* Neutral

The dataset was preprocessed using normalization and transformed into train, validation, and test partitions.

---

## Model Architecture

A custom Fully Convolutional Neural Network (CNN) was designed with the following characteristics:

### Architecture Features

* Two convolutional layers
* Channel sizes: `[32, 64]`
* Residual skip connections
* ReLU activation functions
* Global Average Pooling (GAP)
* 1×1 convolution for classification
* No fully connected (FC) layers

The architecture was designed to remain lightweight while maintaining strong feature extraction capability.

---

## Training Details

### Optimizer

* Adam Optimizer

### Learning Rate

* `1e-3`

### Loss Function

* Cross Entropy Loss

### Training Features

* Validation monitoring
* Hyperparameter tuning
* Loss curve visualization
* Performance tracking across epochs

---

## Evaluation Metrics

The trained model was evaluated using:

* Accuracy
* Precision
* Recall
* Confusion Matrix

These metrics helped analyze the classification performance across different facial expression classes.

---

## Grad-CAM Visualization

Grad-CAM (Gradient-weighted Class Activation Mapping) was implemented to visualize important facial regions influencing model predictions.

The activation maps provide interpretability by highlighting regions responsible for emotion classification decisions.

---

## Adversarial Attack Implementation

A targeted **Fast Gradient Sign Method (FGSM)** adversarial attack was implemented.

### Attack Objective

Modify images originally classified as:

```bash id="sv0fx5"
Sad → Happy
```

The attack evaluates the vulnerability of the CNN model to adversarial perturbations and measures the success rate of targeted misclassification.

---

## Project Structure

```bash id="4ryr64"
├── assignment_2.ipynb          # Jupyter Notebook implementation
├── assignment_2.py             # Python script implementation
├── models/                     # Saved trained models
├── plots/                      # Loss curves and visualizations
├── gradcam_outputs/            # Grad-CAM visualizations
├── adversarial_examples/       # FGSM attack outputs
├── README.md                   # Project documentation
```

---

## How to Run

### Clone the Repository

```bash id="sljlwm"
git clone <your-repository-link>
cd <repository-folder>
```

### Install Dependencies

```bash id="dtx3z5"
pip install torch torchvision numpy matplotlib scikit-learn opencv-python
```

### Run the Notebook

```bash id="6iz2fh"
jupyter notebook
```

Open:

```bash id="0r7hcu"
assignment_2.ipynb
```

---

## Results

The model successfully learned discriminative facial features for emotion classification and achieved stable convergence during training.

Key outcomes:

* Residual CNN improved feature learning efficiency
* Grad-CAM visualizations provided model interpretability
* FGSM attack demonstrated adversarial vulnerability of CNN-based classifiers
* Fully convolutional design reduced parameter complexity

---

## Learning Outcomes

This project helped in understanding:

* CNN architecture design
* Residual learning
* Fully convolutional networks
* Emotion recognition systems
* Grad-CAM interpretability
* Adversarial machine learning
* FGSM attack implementation
* Classification model evaluation

---

## Assignment Reference

This implementation follows the requirements specified in the AI61002 Assignment 2 guidelines.

---

## Author

Mahesh Kusireddy

AI61002 - Deep Learning Foundations and Applications
Indian Institute of Technology Kharagpur
