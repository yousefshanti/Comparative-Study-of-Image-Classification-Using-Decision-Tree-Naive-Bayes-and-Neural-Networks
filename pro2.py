
# Step 1: Prepare Dataset
from PIL import Image
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load and process images
x_data = []
y_data = []
size = (64, 64)  # Image size
folder = "dataset"
total_loaded = 0

for class_idx, class_name in enumerate(sorted(os.listdir(folder))):
    class_path = os.path.join(folder, class_name)
    if not os.path.isdir(class_path):
        continue

    print(f"\nLoading class '{class_name}' (label {class_idx})")

    for img_file in sorted(os.listdir(class_path))[:500]:
        if not img_file.lower().endswith(('.jpeg', '.jpg', '.png')):
            continue
        try:
            img_path = os.path.join(class_path, img_file)
            img = Image.open(img_path)
            img = img.resize(size).convert('L')  # Convert to grayscale
            img_array = np.array(img).flatten() / 255.0  # Normalize
            x_data.append(img_array)
            y_data.append(class_idx)
            total_loaded += 1
        except Exception as e:
            print(f"Error loading image {img_file}: {e}")

print("\nFinished loading images.")
print(f"Total images loaded: {total_loaded}")
print(f"Number of classes: {len(set(y_data))}")

x_data = np.array(x_data)
y_data = np.array(y_data)

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.2, random_state=45, stratify=y_data
)

# Step 1.5: PCA (for Naive Bayes and Decision Tree)
pca = PCA(n_components=50)  # Reduced from 100 to limit complexity
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)

# Step 2: Train Models
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

print("\nTraining models...")

# Model 1: Naive Bayes (on PCA)
m1 = GaussianNB()
m1.fit(x_train_pca, y_train)
r1 = m1.predict(x_test_pca)

# Model 2: Decision Tree (on PCA) - Re-tuned parameters
m2 = DecisionTreeClassifier(
    max_depth=35,              # Increased depth
    min_samples_split=4,       # More flexible splits
    min_samples_leaf=2,        # Prevent very small leaves
    criterion='entropy',
    random_state=45
)
m2.fit(x_train_pca, y_train)
r2 = m2.predict(x_test_pca)

# Model 3: Neural Network (on scaled original data)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

m3 = MLPClassifier(
    hidden_layer_sizes=(1024, 512, 256),  # Increased depth
    activation='relu',
    solver='adam',
    alpha=5e-5,
    learning_rate_init=0.0005,
    max_iter=2000,
    early_stopping=True,
    n_iter_no_change=20,
    validation_fraction=0.1,
    random_state=45
)

print("Training MLP (Neural Network)...")
m3.fit(x_train_scaled, y_train)
r3 = m3.predict(x_test_scaled)

# Step 3: Evaluate Results
from sklearn.metrics import accuracy_score, confusion_matrix

a1 = accuracy_score(y_test, r1)
a2 = accuracy_score(y_test, r2)
a3 = accuracy_score(y_test, r3)

print("\nAccuracy Results:")
print(f"Naive Bayes Accuracy: {a1:.2f}")
print(f"Decision Tree Accuracy: {a2:.2f}")
print(f"Neural Network Accuracy: {a3:.2f}")

cm = confusion_matrix(y_test, r3)
print("\nConfusion Matrix (Neural Network):")
print(cm)

# Step 4: Visualize the first 3 layers of the Decision Tree
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(20, 10))
plot_tree(
    m2,
    max_depth=3,
    filled=True,
    fontsize=10,
    feature_names=[f'PC{i}' for i in range(x_train_pca.shape[1])],
    class_names=[str(c) for c in sorted(set(y_data))]
)
plt.title("Decision Tree - First 3 Layers")
plt.show()