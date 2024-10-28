Thank you for clarifying the noise injection process. Here’s the corrected README with the precise explanation that **noise injection** involves introducing samples from the opposite class, and that **inverse sampling** increases the complexity of data distribution, making prediction more challenging.

---

# Improved Noise-Filtering Algorithm for AdaBoost Using Sample Weight-Based Noise Detection in Imbalanced Datasets

This repository provides an implementation of a noise-filtering algorithm for AdaBoost, developed to improve classification accuracy in imbalanced datasets. By setting specific sample weight-based thresholds, the algorithm distinguishes between noise and essential minority samples, enhancing AdaBoost’s robustness in challenging, imbalanced data contexts.

## Table of Contents
- [Overview](#overview)
- [Data](#data)
- [Methodology](#methodology)
- [Results](#results)
- [Contributing](#contributing)
- [Citation](#citation)

## Overview
In imbalanced datasets, classifiers like AdaBoost often misclassify minority samples as noise, which impacts overall performance. This repository introduces a noise-filtering algorithm with distinct sample weight-based thresholds using absolute and modified SIQR-based criteria. The experimental design includes tests on the original dataset and noise-injected datasets, created with **direct and inverse sampling adjustments** to evaluate the effectiveness of noise detection and prediction performance.

### Key Features:
- **Weight-Based Noise Filtering**: Employs sample weight thresholds with absolute and modified SIQR-based criteria.
- **Noise Injection via Direct and Inverse Sampling**: Introduces noise by adding opposite class samples to specific regions based on dnn values.
- **Improved Generalization**: Limits over-filtering of minority samples, improving AdaBoost performance on imbalanced data.

## Data
The algorithm was validated on multiple imbalanced datasets, showing consistent performance across different imbalance ratios and data distributions.

## Methodology

### Case 1 and Case 2: Absolute and Modified SIQR-Based Weighting Strategies
The noise-filtering algorithm establishes noise thresholds through two main approaches:

1. **Case 1: Absolute Threshold-Based Noise Filtering**  
   In this approach, a fixed threshold is applied independently to each sample weight. Noise is determined solely by whether a sample’s weight exceeds this absolute threshold, making it effective for straightforward noise filtering without considering other samples.

2. **Case 2: Modified SIQR-Based Noise Filtering**  
   This approach uses a modified form of the semi-interquartile range (SIQR) to determine adaptive noise thresholds. The traditional SIQR is adjusted to consider the skewness of sample weight distribution. 

### Experimental Design: Noise Injection via Direct and Inverse Sampling Based on dnn
The experimental setup includes baseline tests on the **original dataset** and noise-injected datasets created by introducing samples from the opposite class based on **direct and inverse sampling** adjustments using **dnn** values:
- **dnn** means ratio of the intra/inter class nearest neighbor distance.
- **Direct Sampling**: Samples closer to the opposite class than to their own class receive higher sampling weights, generating noise samples around these boundary samples. It is likely to generate overlapping region near the boundary area. 
- **Inverse Sampling**: In this approach, samples closer to their own class receive higher weights, injecting opposite-class noise into areas with safe area. This added noise complexity in safe regions makes the data distribution more challenging for prediction.

## Results

### The Difference of the Amount of Noise Detected samples.
In the case of ORBoost, the number of minority samples defined as noise significantly decreases, instead the number of majority samples defined as noise increases.
However, for the case of RUSBostWO, the degree of noise detection changes becomes lower than the degree of ORBoost, because some of majority class samples are deleted before applying noise filtering method.

### Prediction Performance on Original and Noise-Injected Data
The prediction experiments on both original and noise-injected datasets confirmed the effectiveness of the algorithm. According to the conclusions:
- **Case 1 vs Case 2** : The prediction performance of using **Case 1** is more improved than **Case 2**.
- **Original Data Performance**: The algorithm improved minority sample retention by reducing misclassification, enhancing overall prediction performance over traditional noise-filtering AdaBoost (ORBoost, RUSBostWO).
- **Noise-Injected Data Performance**: In the case of using both **Direct** and **Inverse**, proposed noise filtering method showed improved prediction performance. 

## Contributing

Contributions are welcome! If you have suggestions or encounter issues, please submit a pull request or open an issue.

## Citation
For a detailed description of the methodology and experimental results, please refer to the original paper. Journal of Intelligent & Fuzzy Systems [10.3233/JIFS-213244]

