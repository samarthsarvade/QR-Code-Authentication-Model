# visualization_dashboard.py
import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Use a modern seaborn style for a clean look
plt.style.use('seaborn-darkgrid')

# Define dataset paths
first_print_path = r"E:\Projects\QR Code Authentication Model\dataset\First_Print"
second_print_path = r"E:\Projects\QR Code Authentication Model\dataset\Second_Print"
cnn_history_path = r"E:\Projects\QR Code Authentication Model\notebooks\cnn_history.pkl"

# Define a maximum display size for sample images
MAX_SAMPLE_WIDTH = 400
MAX_SAMPLE_HEIGHT = 400

def resize_image(img, max_w=MAX_SAMPLE_WIDTH, max_h=MAX_SAMPLE_HEIGHT):
    """
    Resize image if it exceeds the specified max width or height.
    Maintains aspect ratio by using INTER_AREA interpolation.
    """
    h, w = img.shape[:2]
    if w > max_w or h > max_h:
        # Compute scaling factor to maintain aspect ratio
        scale_w = max_w / w
        scale_h = max_h / h
        scale = min(scale_w, scale_h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img

def load_sample_images(folder, num_samples=1):
    """
    Load up to 'num_samples' images from the specified folder.
    Resizes each image for consistent display.
    Returns a list of RGB images.
    """
    valid_exts = ('.png', '.jpg', '.jpeg')
    image_files = [
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith(valid_exts)
    ]
    samples = []
    for i in range(min(num_samples, len(image_files))):
        img = cv2.imread(image_files[i])
        if img is None:
            continue
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = resize_image(img)
        samples.append(img)
    return samples

# Load sample images (one from each class)
samples_first = load_sample_images(first_print_path, num_samples=1)
samples_second = load_sample_images(second_print_path, num_samples=1)

# Load CNN training history
with open(cnn_history_path, "rb") as f:
    history = pickle.load(f)

def get_image_heights(folder):
    """
    Load each image, resize it, and record the resulting height
    for height distribution analysis.
    """
    heights = []
    valid_exts = ('.png', '.jpg', '.jpeg')
    for file in os.listdir(folder):
        if file.lower().endswith(valid_exts):
            path = os.path.join(folder, file)
            img = cv2.imread(path)
            if img is None:
                continue
            # Convert and resize
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = resize_image(img)
            heights.append(img.shape[0])
    return heights

heights_first = get_image_heights(first_print_path)
heights_second = get_image_heights(second_print_path)

# Create a 2x2 grid layout
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("QR Code Authentication Model - Visualization Dashboard", fontsize=18, y=0.98)

# --- Subplot 1: Sample Image (First Print) ---
axs[0, 0].set_title("Sample Original (First Print)\n(High-quality QR code print)", fontsize=12)
if samples_first:
    axs[0, 0].imshow(samples_first[0])
axs[0, 0].axis('off')

# --- Subplot 2: Sample Image (Second Print) ---
axs[0, 1].set_title("Sample Counterfeit (Second Print)\n(Artifacts from reprinting)", fontsize=12)
if samples_second:
    axs[0, 1].imshow(samples_second[0])
axs[0, 1].axis('off')

# --- Subplot 3: Height Distribution ---
axs[1, 0].set_title("Image Height Distribution\n(Similar heights => uniform preprocessing)", fontsize=12)
sns.kdeplot(heights_first, label="First Prints", ax=axs[1, 0], fill=True, alpha=0.5)
sns.kdeplot(heights_second, label="Second Prints", ax=axs[1, 0], fill=True, alpha=0.5)
axs[1, 0].set_xlabel("Height (pixels)", fontsize=10)
axs[1, 0].set_ylabel("Density", fontsize=10)
axs[1, 0].legend(fontsize=10)

# --- Subplot 4: CNN Training Metrics ---
axs[1, 1].set_title("CNN Training Metrics\n(Accuracy & Loss vs. Epochs)", fontsize=12)
epochs_range = range(1, len(history['accuracy']) + 1)
axs[1, 1].plot(epochs_range, history['accuracy'], label='Train Accuracy', marker='o')
axs[1, 1].plot(epochs_range, history['val_accuracy'], label='Val Accuracy', marker='o')
axs[1, 1].plot(epochs_range, history['loss'], label='Train Loss', marker='o')
axs[1, 1].plot(epochs_range, history['val_loss'], label='Val Loss', marker='o')
axs[1, 1].set_xlabel("Epoch", fontsize=10)
axs[1, 1].set_ylabel("Metric Value", fontsize=10)
axs[1, 1].legend(fontsize=10)

# Adjust layout for a clean look
plt.tight_layout(pad=2.0, rect=[0, 0.03, 1, 0.95])

# Optionally, try to maximize the window (Windows + TkAgg backend)
try:
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
except Exception:
    pass

plt.show()
