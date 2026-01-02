# EE4065 Homework 5 - Embedded AI Applications

**Course:** EE 4065 - Embedded Digital Image Processing
**Due Date:** Jan 2, 2026
**Student:** [Adın Soyadın] - [Öğrenci Numaran]

## Project Description
This repository implements **Keyword Spotting (KWS)** and **Handwritten Digit Recognition (MNIST)** applications on STM32. As per the homework requirements, **offline datasets** (static C arrays) are used to validate model inference on the microcontroller without external sensors.

---

## Q1: Keyword Spotting (Section 12.8)

**Method:** CNN with **Polynomial Decay Pruning** (50-80% sparsity) to optimize Flash usage.
**Input:** Simulated MFCC input buffer via `offline_data.h`.
**Result:** Successful detection of "YES" keyword with high confidence.

### Q1 Inference Output:
<img width="520" alt="Q1 Result" src="https://github.com/user-attachments/assets/7d13cb60-6571-46b2-bf41-fff39cff2452" />

---

## Q2: Handwritten Digit Recognition (Section 12.9)

**Method:** CNN trained on MNIST and converted using **Dynamic Range Quantization** (Int8).
**Input:** A digit '7' sample extracted from MNIST test set via `digit_data.h`.
**Result:** ~98% accuracy on validation set.

### Q2 Training & Validation Output:
<img width="1491" alt="Q2 Result" src="https://github.com/user-attachments/assets/4b7d3cd9-a7c0-45dd-98f3-d0fb153c77f4" />

---

**Reference:**
C. Ünsalan, B. Höke, and E. Atmaca, *Embedded Machine Learning with Microcontrollers: Applications on STM32 Boards*, Springer Nature, 2025.
