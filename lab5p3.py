# ---- Paths (reuse) ----
import os, json, numpy as np, torch
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
from sklearn.model_selection import train_test_split

DATASET_ROOT = r"/Users/michaelnguyen/IdeaProjects/ProjectTwo/Knn-Classifier-Oxford-Flower"
DATASET_PATH = os.path.join(DATASET_ROOT, "dataset")
CAT_TO_NAME_PATH = os.path.join(DATASET_ROOT, "cat_to_name.json")

with open(CAT_TO_NAME_PATH, "r") as f:
    cat_to_name = json.load(f)

# ---- Loader (reuse) ----
def load_flower_dataset(dataset_path, cat_to_name_dict, max_classes=None, min_images=100):
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
        cls = cat_dir_to_idx[cat_dir]
        files = [f for f in os.listdir(cat_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        for f in files:
            p = os.path.join(cat_path, f)
            try:
                img = imread(p)
                if img.ndim == 3: img = rgb2gray(img)
                if img.ndim != 2: continue
                images.append(img)
                labels.append(cls)
            except Exception:
                continue
    return np.array(images, dtype=object), np.array(labels), class_names

# ---- Preprocess (reuse; outputs 4D for CNN) ----
TARGET_SIZE = 64

def preprocess_to_cnn(images):
    """Resize to 64x64, scale to [0,1], add channel dim => (N,1,64,64)."""
    N = len(images)
    out = np.empty((N, 1, TARGET_SIZE, TARGET_SIZE), dtype=np.float32)
    for i, img in enumerate(images):
        r = resize(img, (TARGET_SIZE, TARGET_SIZE), anti_aliasing=True).astype(np.float32)
        out[i, 0] = r
    return out

# ---- Load + split (reuse) ----
MAX_CLASSES = 10
MIN_IMAGES_PER_CLASS = 100

X_all_raw, y_all, class_names = load_flower_dataset(
    DATASET_PATH, cat_to_name, max_classes=MAX_CLASSES, min_images=MIN_IMAGES_PER_CLASS
)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_all_raw, y_all, test_size=0.20, random_state=42, stratify=y_all
)

X_train = preprocess_to_cnn(X_train_raw)   # (N,1,64,64)
X_test  = preprocess_to_cnn(X_test_raw)

# ---- Torch Datasets (reuse) ----
class NumpyTensorDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)                  # float32
        self.y = torch.from_numpy(y).long()           # int64
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

train_DS = NumpyTensorDataset(X_train, y_train)
test_DS  = NumpyTensorDataset(X_test,  y_test)

num_classes = len(class_names)
device = "cuda" if torch.cuda.is_available() else "cpu"
