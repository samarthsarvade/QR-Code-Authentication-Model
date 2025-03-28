# QR Code Authentication Model

QR Code Authentication: Detecting Original vs. Counterfeit Prints 🤓
Developed a hybrid ML model combining traditional computer vision (HOG + SVM) with deep learning (CNN) to distinguish genuine from counterfeit QR codes. Utilized TensorFlow 2.10, scikit-learn, OpenCV, and scikit-image for image preprocessing, augmentation, and feature extraction😵‍💫. The CNN achieved 95.54% validation accuracy compared to 74% for the SVM, demonstrating resilience against scanning artifacts. 👀 Learned to integrate classical methods with deep learning and leverage GPU acceleration (where CUDA and cuDNN 😶‍🌫️ bother) for efficient model training.

This project detects original (first print) versus counterfeit (second print) QR code images using two approaches:
1. A traditional computer vision pipeline (HOG feature extraction with SVM).
2. A deep learning-based approach (CNN with data augmentation).

## Project Structure

- **dataset/**  
  Contains `First_Print` (originals) and `Second_Print` (counterfeits).

- **notebooks/**  
  Contains interactive notebooks for data exploration (`data_exploration.ipynb`) and CNN training/visualization (`cnn_training.ipynb`).

- **src/**  
  Contains Python scripts:
  - `traditional_pipeline.py`: SVM training using HOG features.
  - `cnn_model.py`: CNN model training.
  - `utils.py`: Utility functions for data loading and visualization.
    
- **Results/**
  ![Figure_1](https://github.com/user-attachments/assets/72568ec2-b66d-42c2-bfcd-dfa0131a6973)

- **requirements.txt**  
  Lists the required Python packages.

## Setup and Execution

1. Create and activate a Python virtual environment.
2. Install the required packages using:
