# Higgs Boson Machine Learning Challenge

This repository contains my work on the Kaggle Higgs Boson Machine Learning Challenge. goal is to classify events from the Higgs Boson dataset into either a signal (a real Higgs Boson event) or background (not a Higgs Boson event).

## Overview

While I explored mayn different approaches, two remained.

### Deep Neural Network (DNN)
   I built a Deep Neural Network using TensorFlow. The DNN was tuned with various architectures, activation functions, and optimizers to improve classification accuracy.
   The extrapolated AMS reached $2.6$.

### Gaussian Mixture Model (GMM)
   In addition to the DNN, I experimented with a Gaussian Mixture Model, which I believe is a relatively novel approach for this challenge. The GMM was used to model the probability distributions of the signal and background events separately. Results were mediocre, reaching an AMS score of $1.7$ for the complete dataset.


---
![fullmodel](/documentation/img/fullmodel.png)
Principal components for signal and background of the training dataset.
