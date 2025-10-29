import os
import json
import numpy as np
import torch
import torch.nn as nn
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
DATASET_ROOT = r"/Users/michaelnguyen/IdeaProjects/ProjectTwo/Knn-Classifier-Oxford-Flower"
DATASET_PATH = os.path.join(DATASET_ROOT, "dataset")
CAT_TO_NAME_PATH = os.path.join(DATASET_ROOT, "cat_to_name.json")

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset not found at: {DATASET_PATH}")
if not os.path.exists(CAT_TO_NAME_PATH):
    raise FileNotFoundError(f"Category names file not found at: {CAT_TO_NAME_PATH}")

with open(CAT_TO_NAME_PATH, 'r') as f:
    cat_to_name = json.load(f)

# ---------------------------------------------------------------------
# Dataset Loader
# ---------------------------------------------------------------------
def load_flower_dataset(dataset_path, cat_to_name_dict, max_classes=None, min_images=100):
    """
    Load flower dataset from flat directory structure
    Returns grayscale images and integer labels.
    """
    images, labels = [], []
    category_info = []

    for d in os.listdir(dataset_path):
        cat_path = os.path.join(dataset_path, d)
        if os.path.isdir(cat_path):
            img_count = len([f for f in os.listdir(cat_path)
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            if img_count >= min_images:
                category_info.append((d, img_count))

    category_info.sort(key=lambda x: x[1], reverse=True)
    if max_classes:
        category_info = category_info[:max_classes]

    category_dirs = sorted([cat_dir for cat_dir, _ in category_info], key=lambda x: int(x))
    cat_dir_to_idx = {cat_dir: idx for idx, cat_dir in enumerate(category_dirs)}
    class_names = [cat_to_name_dict.get(cat_dir, f"Unknown_{cat_dir}") for cat_dir in category_dirs]

    for cat_dir in category_dirs:
        cat_path = os.path.join(dataset_path, cat_dir)
        class_idx = cat_dir_to_idx[cat_dir]
        files = [f for f in os.listdir(cat_path)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        for f in files:
            img_path = os.path.join(cat_path, f)
            try:
                img = imread(img_path)
                if img.ndim == 3:
                    img = rgb2gray(img)
                elif img.ndim != 2:
                    continue
                images.append(img)
                labels.append(class_idx)
            except Exception:
                continue

    return np.array(images, dtype=object), np.array(labels), class_names

# ---------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------
TARGET_SIZE = 64

def preprocess_images(images):
    """Resize to 64x64 and normalize to [0,1]."""
    out = np.empty((len(images), TARGET_SIZE, TARGET_SIZE), dtype=np.float32)
    for i, img in enumerate(images):
        out[i] = resize(img, (TARGET_SIZE, TARGET_SIZE), anti_aliasing=True).astype(np.float32)
    return out

# ---------------------------------------------------------------------
# Load and Split
# ---------------------------------------------------------------------
MAX_CLASSES = 10
MIN_IMAGES_PER_CLASS = 100

print(f"Loading dataset (Top {MAX_CLASSES} classes with {MIN_IMAGES_PER_CLASS}+ images)...")
X_all_raw, y_all, class_names = load_flower_dataset(
    DATASET_PATH, cat_to_name, max_classes=MAX_CLASSES, min_images=MIN_IMAGES_PER_CLASS
)

print(f"Loaded {len(X_all_raw)} images across {len(class_names)} classes")

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_all_raw, y_all, test_size=0.2, random_state=42, stratify=y_all
)

print("Resizing and normalizing images...")
X_train_resized = preprocess_images(X_train_raw)
X_test_resized  = preprocess_images(X_test_raw)

print("Flattening images for model input...")
X_train_flat = X_train_resized.reshape(len(X_train_resized), -1)
X_test_flat  = X_test_resized.reshape(len(X_test_resized), -1)

X_train_t = torch.tensor(X_train_flat, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test_flat, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

train_DS = torch.utils.data.TensorDataset(X_train_t, y_train_t)
test_DS = torch.utils.data.TensorDataset(X_test_t, y_test_t)

input_dim = X_train_t.shape[1]
num_classes = len(class_names)
# ---------------------------------------------------------------------
# Configurations (6 runs)
# ---------------------------------------------------------------------
configs = [
    {"id": 1, "hidden_layers": [128], "lr": 1e-3, "batch": 32, "epochs": 10},
    {"id": 2, "hidden_layers": [128, 64], "lr": 1e-3, "batch": 32, "epochs": 10},
    {"id": 3, "hidden_layers": [256, 128], "lr": 1e-3, "batch": 32, "epochs": 10},
    {"id": 4, "hidden_layers": [512, 256, 128], "lr": 1e-3, "batch": 32, "epochs": 10},
    {"id": 5, "hidden_layers": [256, 128, 64], "lr": 1e-3, "batch": 32, "epochs": 12},
    {"id": 6, "hidden_layers": [256, 128, 128], "lr": 1e-2, "batch": 32, "epochs": 10},
]

class MLP(nn.Module):
    def __init__(self, input_dim, layer_sizes, num_classes):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in layer_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

    
    
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------
# Run experiments for all configs
# ---------------------------------------------------------------------
results = []

for cfg in configs:
    print(f"\n=== Running Config {cfg['id']} ===")
    print(f"Layers: {cfg['hidden_layers']}, LR: {cfg['lr']}, Batch: {cfg['batch']}, Epochs: {cfg['epochs']}")

    trainloader = torch.utils.data.DataLoader(train_DS, batch_size=cfg["batch"], shuffle=True)
    testloader = torch.utils.data.DataLoader(test_DS, batch_size=cfg["batch"], shuffle=False)

    # make new model + optimizer per run
    model = MLP(input_dim, cfg["hidden_layers"], num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"])

    losses = []

    for epoch in range(cfg["epochs"]):
        temp_loss = []
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            temp_loss.append(loss.item())
        losses.append(np.mean(temp_loss))
        print(f"Epoch {epoch+1}/{cfg['epochs']} - Loss: {losses[-1]:.4f}")

    # compute accuracy
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(f"Test Accuracy: {acc:.2f}%")

    results.append({
        "id": cfg["id"],
        "layers": cfg["hidden_layers"],
        "lr": cfg["lr"],
        "batch": cfg["batch"],
        "epochs": cfg["epochs"],
        "final_loss": losses[-1],
        "test_acc": acc,
        "losses": losses
    })

print("\n=== Summary of All Runs ===")
for r in results:
    print(f"Config {r['id']} | Layers={r['layers']} | LR={r['lr']} | "
          f"Batch={r['batch']} | Acc={r['test_acc']:.2f}% | Final Loss={r['final_loss']:.4f}")

print(f"Train set: {len(X_train_resized)} | Test set: {len(X_test_resized)}")
print(f"Image shape: {X_train_resized.shape[1:]} | Classes: {len(class_names)}")

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Identify and print best configuration
# ---------------------------------------------------------------------
best_run = max(results, key=lambda x: x["test_acc"])
print("\n=== Best Configuration ===")
print(f"Config {best_run['id']}: Layers={best_run['layers']}, LR={best_run['lr']}, "
      f"Batch={best_run['batch']}, Epochs={best_run['epochs']}")
print(f"Test Accuracy={best_run['test_acc']:.2f}% | Final Loss={best_run['final_loss']:.4f}")
print("Justification: This configuration likely achieved the best accuracy because its layer "
      "sizes and learning rate allowed stable learning without overfitting.")

# ---------------------------------------------------------------------
# Plot 1: Accuracy vs. Configuration
# ---------------------------------------------------------------------
plt.figure(figsize=(7,4))
plt.plot([r["id"] for r in results], [r["test_acc"] for r in results], marker='o')
plt.title("Accuracy vs Configuration Index")
plt.xlabel("Configuration ID")
plt.ylabel("Test Accuracy (%)")
plt.grid(True, alpha=0.4)
plt.savefig("accuracy_vs_config.png", dpi=150, bbox_inches='tight')
plt.show()

# ---------------------------------------------------------------------
# Plot 2: Loss vs. Epoch for two representative runs (Config 1 and best)
# ---------------------------------------------------------------------
plt.figure(figsize=(8,4))
for r in results:
    if r["id"] in [1, best_run["id"]]:
        plt.plot(r["losses"], label=f"Config {r['id']}")
plt.title("Loss vs Epoch (Representative Runs)")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.legend()
plt.grid(True, alpha=0.4)
plt.savefig("loss_vs_epoch.png", dpi=150, bbox_inches='tight')
plt.show()
