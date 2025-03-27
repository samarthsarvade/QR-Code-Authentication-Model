# src/utils.py
import os
import cv2
import matplotlib.pyplot as plt

def load_image_paths(folder, extensions=('.png', '.jpg', '.jpeg')):
    """
    Return a list of image paths in a given folder filtered by specified extensions.
    """
    return [os.path.join(folder, file) for file in os.listdir(folder) if file.lower().endswith(extensions)]

def load_and_preprocess_image(image_path, size=(128, 128), color_mode='grayscale'):
    """
    Load an image from disk, resize it, and convert its color space.
    
    :param image_path: Path to the image file.
    :param size: Tuple of (width, height) to resize the image.
    :param color_mode: 'grayscale' or 'rgb'
    :return: Preprocessed image.
    """
    if color_mode == 'grayscale':
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image = cv2.resize(image, size)
    return image

def plot_sample_images(image_paths, title="Sample Images", num_samples=4):
    """
    Plot a set of sample images from a list of image paths.
    """
    plt.figure(figsize=(10, 10))
    for i, img_path in enumerate(image_paths[:num_samples]):
        img = load_and_preprocess_image(img_path, size=(128, 128), color_mode='rgb')
        plt.subplot(1, num_samples, i+1)
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Example usage:
    sample_folder = r"E:\Projects\QR Code Authentication Model\dataset\First_Print"
    paths = load_image_paths(sample_folder)
    print(f"Found {len(paths)} images in {sample_folder}")
    plot_sample_images(paths, title="First Print Samples")
