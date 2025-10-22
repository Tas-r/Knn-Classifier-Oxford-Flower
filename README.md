# 102 Category Flower Dataset

<h3>Maria-Elena Nilsback and Andrew Zisserman</h3>

## Project: K-NN Flower Classifier with HOG Features

This repository implements a **K-Nearest Neighbors (K-NN) classifier** for flower species classification using **Histogram of Oriented Gradients (HOG)** features on the Oxford 102 Flower Dataset.

### Implementation Overview

**Objective:** Build a traditional computer vision pipeline to classify flower images into 102 categories using handcrafted features and a K-NN classifier.

**Pipeline:**

1. **Dataset Preprocessing:** Load flower images from train/valid splits, convert to grayscale, resize to 64×64 pixels
2. **Feature Extraction:** Extract HOG features (1,764-dimensional vectors) to capture shape and edge information
3. **Classification:** Train K-NN classifier with hyperparameter tuning (K sweep from 1 to 21)
4. **Evaluation:** Analyze performance with confusion matrices, classification reports, and per-class metrics

### Key Features

- ✅ **HOG Feature Extraction:** Using scikit-image library with 9 orientation bins, 8×8 pixel cells, and L2-Hys normalization
- ✅ **K-NN Classifier:** Configurable K values with parallel processing for efficient training
- ✅ **Hyperparameter Tuning:** Systematic K sweep to find optimal value (K=3-5 typically best)
- ✅ **Comprehensive Evaluation:** Classification reports, confusion matrices, and per-class accuracy analysis
- ✅ **Flexible Configuration:** Run with 20 classes (quick testing) or all 102 classes (full evaluation)

### Results

**Configuration:** 20 flower categories, 903 training images, 130 test images

**Performance:**

- Best K value: **3**
- Training accuracy: **55.26%**
- Test accuracy: **9.23%**
- Feature vector size: **1,764 dimensions**

**Key Findings:**

- HOG features effectively capture shape/edge information but struggle with fine-grained differences
- Grayscale conversion loses critical color information needed for flower classification
- K-NN shows significant overfitting (large accuracy gap between train and test)
- Best performing categories: hard-leaved pocket orchid (100%), pink primrose (25%)
- Most confused pairs: yellow iris → purple coneflower (10 errors)

### Limitations & Future Work

**Current Limitations:**

- Grayscale processing loses color information (crucial for flowers)
- HOG features don't capture fine-grained texture details
- K-NN scalability issues with large datasets
- High-dimensional feature space (1,764 dimensions)

**Proposed Improvements:**

1. **Color Features:** Add RGB/HSV histograms to capture color information
2. **Multi-scale HOG:** Extract features at different scales for better detail capture
3. **Dimensionality Reduction:** Apply PCA to reduce feature space
4. **Advanced Classifiers:** Try SVM with RBF kernel or ensemble methods
5. **Deep Learning:** CNNs with transfer learning can achieve 80-95% accuracy vs. current 9-55%

### Technical Stack

- **Python 3.x**
- **NumPy:** Numerical operations
- **scikit-image:** HOG feature extraction
- **scikit-learn:** K-NN classifier and evaluation metrics
- **Matplotlib/Seaborn:** Visualization
- **Jupyter Notebook:** Interactive development

### Dataset Details

We have created a 102 category dataset, consisting of 102 flower categories. The flowers chosen to be flower commonly occuring in the United Kingdom. Each class consists of between 40 and 258 images. The details of the categories and the number of images for each class can be found on this category statistics page.

The images have large scale, pose and light variations. In addition, there are categories that have large variations within the category and several very similar categories. The dataset is visualized using isomap with shape and colour features.

## Directory Structure

```
> dataset
	> train
	> valid
	> test
- cat_to_name.json
- README.md
- sample_submission.csv
```

## Visualization of the dataset

We visualize the categories in the dataset using SIFT features as shape descriptors and HSV as colour descriptor. The images are randomly sampled from the category.

![](https://i.imgur.com/Tl6TKUC.png)

## Publications

Nilsback, M-E. and Zisserman, A.
<a href="https://www.robots.ox.ac.uk/~vgg/publications/2008/Nilsback08/">Automated flower classification over a large number of classes</a>  
Proceedings of the Indian Conference on Computer Vision, Graphics and Image Processing (2008)

## Source

- Original source of this data can be found in <a href="https://www.robots.ox.ac.uk/~vgg/data/flowers/102/"> 102 Category Flower Dataset</a>
- Original readme from author can be found in <a href="https://www.robots.ox.ac.uk/~vgg/data/flowers/102/README.txt">AUTHOR README</a>
- Directory test is added from another kaggle dataset that can be found in <a href="https://www.kaggle.com/c/oxford-102-flower-pytorch/">Oxford 102 Flower Pytorch</a>
