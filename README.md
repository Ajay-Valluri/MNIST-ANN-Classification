# MNIST Handwritten Digit Classification (PyTorch)

This project develops an Artificial Neural Network (ANN) and uses PyTorch to classify handwritten digits from the MNIST dataset. The model was trained, validated, and tested while the generalization performance was improved by fine-tuning multiple hyperparameters. Although the model has a simple architecture, the ANN achieved strong accuracy on MNIST through careful optimization and regularization strategies. 

---

## Project Overview

- **Task:** Image classification (10 classes: digits 0–9)
- **Dataset:** MNIST (70,000 grayscale images, 28×28 px)
- **Model Type:** Fully connected neural network (ANN)
- **Framework:** PyTorch
- **Performance:** ~98.24% validation accuracy

This work demonstrates a foundational computer vision pipeline and highlights how classical ANN architectures can still perform competitively on simple datasets with modern training practices.

---

## Features

- Data preprocessing with normalization  
- Train/validation/test split  
- ANN with 4 fully connected layers  
- ReLU activations  
- Dropout + weight decay regularization  
- Learning rate scheduling  
- Training + validation accuracy & loss tracking  
- Final evaluation on test set  
- Training metrics visualized over epochs  

---

## Model Architecture

| Layer | Type | Dimensions |
|-------|------|------------|
| Input | Flatten | 28×28 → 784 |
| FC1   | Linear + ReLU | 784 → 512 |
| FC2   | Linear + ReLU | 512 → 256 |
| FC3   | Linear + ReLU | 256 → 128 |
| FC4   | Linear | 128 → 10 |

Additional techniques:
- Dropout applied between layers
- Weight decay used for regularization
- Learning-rate scheduler (factor=0.5, patience=3)

---

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Epochs | 25 |
| Batch Size | 128 |
| Learning Rate | 5e-4 |
| Weight Decay | 5e-5 |
| Dropout | 0.4 → 0.3 → 0.3 |
| Optimizer | Adam |
| Loss Function | CrossEntropyLoss |
| Scheduler | ReduceLROnPlateau |

Hyperparameter search included grid-based tuning and iterative refinement.

---

## Results

Plots generated:
- Training vs. Validation Loss
- Training vs. Validation Accuracy

Epoch-level metrics and key results were saved to a CSV file.
