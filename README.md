# 102 Category Flower Dataset

<h3>Maria-Elena Nilsback and Andrew Zisserman</h3>

## Project: K-NN Flower Classifier with HOG Features

This repository implements a **K-Nearest Neighbors (K-NN) classifier** for flower species classification using **Histogram of Oriented Gradients (HOG)** features on the Oxford 102 Flower Dataset.

### Implementation Overview

**Objective:** Build a traditional computer vision pipeline to classify flower images into 102 categories using handcrafted features and a K-NN classifier.

**Pipeline:**

1. **Dataset Loading:** Load all flower images from flat directory structure (folders 1-102)
2. **Data Splitting:** Split dataset 80:20 for training and testing using stratified sampling
3. **Preprocessing:** Convert to grayscale, resize to 64×64 pixels, normalize to [0,1]
4. **Feature Extraction:** Extract HOG features (1,764-dimensional vectors) to capture shape and edge information
5. **Classification:** Train K-NN classifier with hyperparameter tuning (K sweep from 1 to 21)
6. **Evaluation:** Analyze performance with confusion matrices, classification reports, and per-class metrics

### Key Features

- ✅ **Smart Class Selection:** Uses top 10 classes with 100+ images each for balanced dataset
- ✅ **Flat Dataset Structure:** All images organized in numbered folders (1-102) for each flower category
- ✅ **Stratified 80:20 Split:** Ensures balanced class distribution in training and test sets
- ✅ **HOG Feature Extraction:** Using scikit-image library with 9 orientation bins, 8×8 pixel cells, and L2-Hys normalization
- ✅ **K-NN Classifier:** Configurable K values with parallel processing for efficient training
- ✅ **Hyperparameter Tuning:** Systematic K sweep to find optimal value (K=3-5 typically best)
- ✅ **Comprehensive Evaluation:** Classification reports, confusion matrices, and per-class accuracy analysis
- ✅ **Flexible Configuration:** Adjustable min_images threshold and max_classes parameters

### Results

**Configuration:**

- **Classes:** 10 flower categories (with 100+ images each)
- **Total images:** ~1,700 images
- **Dataset split:** 80% training (~1,360 images), 20% testing (~340 images)
- **Feature vector size:** 1,764 dimensions (HOG)
- **Training method:** Stratified split ensures balanced class distribution

**Performance (based on 80:20 split):**

- Best K value determined through cross-validation sweep (K=1,3,5,7,9,11)
- Using classes with sufficient data (100+ images) improves model reliability
- Results demonstrate effectiveness of traditional CV methods on balanced datasets

**Key Findings:**

- HOG features effectively capture shape/edge information but struggle with fine-grained differences
- Grayscale conversion loses critical color information needed for flower classification
- K-NN performance depends on optimal K selection (typically K=3-5)
- Fine-grained flower classification is challenging with traditional CV methods
- Color information is crucial for distinguishing similar flower species

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
Knn-Classifier-Oxford-Flower/
├── dataset/
│   ├── 1/          # Pink primrose images
│   ├── 2/          # Hard-leaved pocket orchid images
│   ├── 3/          # Canterbury bells images
│   └── ...         # 102 total category folders (numbered 1-102)
├── cat_to_name.json    # Mapping of category ID to flower name
├── lab5.ipynb          # Main implementation notebook
├── README.md
└── sample_submission.csv
```

**Dataset Organization:**

- Each numbered folder (1-102) contains all images for that flower category
- Images are loaded and split 80:20 for training/testing using stratified sampling
- Total of **6,552 images** across 102 flower species

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
