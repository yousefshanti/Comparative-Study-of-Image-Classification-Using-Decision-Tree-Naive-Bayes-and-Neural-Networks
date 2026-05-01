# 🧠 Image Classification: Decision Tree, Naive Bayes & Feedforward Neural Networks

> **ENCS3340 – Artificial Intelligence** | Faculty of Engineering & Technology, Electrical & Computer Engineering Department  
> **Birzeit University**

---

## 👥 Authors

| Name | Student Number | Section |
|------|---------------|---------|
| Yousef Shanti | 1221137 | 2 |

**Instructors:** Yazan Abu Farha & Samah Alaydi  
**Date:** 28 / 6 / 2025

---

## 📌 Overview

This project presents a **comparative study** of three classical machine learning models applied to **grayscale image classification**:

- 🌿 **Naive Bayes** (GaussianNB)
- 🌳 **Decision Tree** (Entropy-based)
- 🤖 **Feedforward Neural Network** (MLPClassifier)

All models are trained on a custom dataset containing images of **birds**, **horses (cavallo)**, and **jellyfish**, and evaluated based on accuracy, interpretability, and learning capacity.

---

## 📂 Dataset Description & Preprocessing

- **Total images:** 704
- **Classes:**
  - 🐦 Bird (Label 1)
  - 🐴 Cavallo / Horse (Label 2)
  - 🪼 Jellyfish (Label 3)

**Preprocessing pipeline:**
1. Converted to **grayscale**
2. Resized to **64×64** pixels
3. Flattened into a **4,096-dimensional vector**
4. **Normalized** (for MLP)
5. **PCA-reduced** to 50 components (for Naive Bayes & Decision Tree)
6. **80/20 train-test split** using stratified sampling

### Dataset Variants

| Dataset | Description |
|---------|-------------|
| **D1** | Balanced small dataset (dog=88, bird=131, cat=166) — all cat images are faces only |
| **D2** | Random images, large in size (bird=139, cat=126, dog=134) |
| **D3** | Similar to D2 with different sampling (bird=138, cat=129, dog=126) |
| **Dall** | Combination of D1, D2, and D3 (bird=321, cat=328, dog=253) |
| **DX1** | Large dataset (horse=2623, cat=1668, dog=4863) — bird class renamed to horse |
| **DX2** | Same as DX1 but cat class renamed to sheep (sheep=1820) |

---

## ⚙️ Model Implementation

### 🔹 Naive Bayes Classifier
- Used `GaussianNB` from **scikit-learn**
- Applied on **PCA-reduced** data
- Chosen for simplicity and baseline performance
- Does **not** require feature scaling

### 🔹 Decision Tree Classifier
- Used `DecisionTreeClassifier` with:
  ```
  criterion='entropy'
  max_depth=35
  min_samples_split=4
  min_samples_leaf=2
  ```
- Applied on **PCA-reduced** data
- Offers hierarchical decisions and clear interpretability

### 🔹 Feedforward Neural Network (MLPClassifier)
- Used `MLPClassifier` with:
  ```
  hidden_layer_sizes=(1024, 512, 256)
  max_iter=2000
  early_stopping=True
  validation_fraction=0.1
  ```
- Trained on **scaled original data**
- Best suited for learning complex, nonlinear relationships

---

## 📊 Results

### Accuracy Comparison

| Model | Accuracy |
|-------|----------|
| 🔹 Naive Bayes | **0.70** |
| 🌳 Decision Tree | **0.70** |
| 🤖 Neural Network (MLP) | **0.83** |

### Confusion Matrix – Neural Network (MLP)

```
Predicted →   Bird   Horse   Jellyfish
Bird           8      15        3
Horse          1      96        3
Jellyfish      0       2       13
```

---

## 🔍 Comparative Analysis

| Model | Strengths | Limitations |
|-------|-----------|-------------|
| **Naive Bayes** | Fast, simple, no scaling needed | Limited by feature independence assumption |
| **Decision Tree** | Interpretable, visualizable structure | Prone to overfitting, sensitive to depth |
| **MLP Neural Network** | Highest accuracy, handles nonlinear patterns | Computationally intensive, harder to tune |

---

## 💬 Discussion

### ✅ Strengths
- **Naive Bayes:** Fast and simple, good for quick baselines
- **Decision Tree:** Easy to visualize and explain decisions
- **MLP:** Best accuracy, capable of handling image complexity

### ⚠️ Challenges Faced
- Tuning MLP hyperparameters
- Visual similarity between classes in grayscale
- Balancing depth in decision trees to avoid overfitting

### 🔮 Future Work
- Try **CNNs** for spatial feature learning
- Use **data augmentation** to improve robustness
- Explore **hyperparameter tuning** (e.g., `GridSearchCV`)
- Use **color images** or additional features (e.g., HOG descriptors)

---

## 📝 Conclusion

This study highlights the balance between **interpretability** and **performance** in image classification.

> While Naive Bayes and Decision Trees offer simplicity and clear visualization, the Neural Network significantly outperforms both in accuracy — showcasing the power of deep learning on image data.

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![NumPy](https://img.shields.io/badge/NumPy-array-lightblue?logo=numpy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-visualization-green)

---

*Faculty of Engineering & Technology – Birzeit University*
