# Bag of Visual Words Image Classifier

This Git repository contains an image classifier that utilizes the Bag of Visual Words (BoVW) approach with three different classification algorithms implemented from scratch: Naive Bayes, Logistic Regression, and a Fully Connected Neural Network.

## Table of Contents

- [Introduction](#introduction)
- [Setup](#setup)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Implementation Details](#implementation-details)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Introduction

The Bag of Visual Words (BoVW) is a popular technique in computer vision for image classification tasks. It involves breaking down an image into smaller regions, extracting local features from these regions, and then creating a histogram of visual words (clusters of features) to represent the image. In this repository, we implement a BoVW-based image classifier with three different classification algorithms:

1. **Naive Bayes**: A probabilistic classification algorithm based on Bayes' theorem, which assumes independence between features.
2. **Logistic Regression**: A linear classification algorithm that models the probability of a sample belonging to a particular class.
3. **Fully Connected Neural Network**: A multi-layer perceptron (MLP) neural network for non-linear classification.

These algorithms are implemented from scratch to provide a deeper understanding of their inner workings. The histograms can be found under data folder.


