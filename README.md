# EE4065 Homework 5 - Embedded AI Applications

**Course:** EE 4065 - Embedded Digital Image Processing
**Due Date:** Jan 2, 2026
**Student:** [Adın Soyadın] - [Öğrenci Numaran]

## Project Description
This repository implements **Keyword Spotting (KWS)** and **Handwritten Digit Recognition (MNIST)** applications on STM32, based on the course textbook (Ünsalan et al., 2025) .

As per the homework requirements, **offline datasets** (static C arrays) are used to validate model inference on the microcontroller without external sensors .

Implementation Details
Q1: Keyword Spotting (Section 12.8) 
<img width="520" height="335" alt="Ekran görüntüsü 2026-01-02 223140" src="https://github.com/user-attachments/assets/7d13cb60-6571-46b2-bf41-fff39cff2452" />

Method: CNN with Polynomial Decay Pruning (50-80% sparsity) to optimize Flash usage.

Validation: Validated using a simulated MFCC input buffer via offline_data.h.

Result: Successful detection of "YES" keyword (See Q1_Inference_Result.png).

Q2: Digit Recognition (Section 12.9) 
<img width="1491" height="95" alt="Ekran görüntüsü 2026-01-02 223715" src="https://github.com/user-attachments/assets/4b7d3cd9-a7c0-45dd-98f3-d0fb153c77f4" />

Method: CNN trained on MNIST and converted using Dynamic Range Quantization (Int8).

Validation: Validated using a digit '7' sample from MNIST test set via digit_data.h.

Result: ~98% accuracy on validation set (See Q2_Training_Result.png).
