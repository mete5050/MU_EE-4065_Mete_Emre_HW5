# EE4065 Homework 5 - Embedded AI Applications

**Course:** EE 4065 - Embedded Digital Image Processing
**Due Date:** Jan 2, 2026
**Student:** [Adın Soyadın] - [Öğrenci Numaran]

## Project Description
This repository implements **Keyword Spotting (KWS)** and **Handwritten Digit Recognition (MNIST)** applications on STM32, based on the course textbook (Ünsalan et al., 2025) .

As per the homework requirements, **offline datasets** (static C arrays) are used to validate model inference on the microcontroller without external sensors .

## Repository Structure
```text
.
├── Python_Training/           # Training scripts & Result Screenshots
│   ├── train_kws.py           # Q1: Training with Pruning
│   ├── train_mnist.py         # Q2: Training with Quantization
│   ├── Q1_Inference_Result.png
│   └── Q2_Training_Result.png
├── STM32_Project/             # STM32CubeIDE Project
│   ├── Core/Src/main.c        # Main application logic
│   ├── Core/Inc/offline_data.h # Q1 Test Data
│   └── Core/Inc/digit_data.h   # Q2 Test Data
└── README.md


Implementation Details
Q1: Keyword Spotting (Section 12.8) 

Method: CNN with Polynomial Decay Pruning (50-80% sparsity) to optimize Flash usage.

Validation: Validated using a simulated MFCC input buffer via offline_data.h.

Result: Successful detection of "YES" keyword (See Q1_Inference_Result.png).

Q2: Digit Recognition (Section 12.9) 

Method: CNN trained on MNIST and converted using Dynamic Range Quantization (Int8).

Validation: Validated using a digit '7' sample from MNIST test set via digit_data.h.

Result: ~98% accuracy on validation set (See Q2_Training_Result.png).
