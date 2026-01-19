# Driver Drowsiness Detection

This repository contains a real-time driver drowsiness detection system built using computer vision and deep learning. The system uses a webcam to monitor the driver's face, detects signs of drowsiness (specifically yawning), and provides an alert. The project includes scripts for data labeling, preprocessing, model training, and a real-time detection application. The project leverages MediaPipe for efficient face detection and MobileNetV2 for classification.

## How It Works

The system operates through a multi-stage pipeline:

1.  **Data Labeling:** The `src/video_labeling.py` script provides a GUI to manually label video frames as either 'normal' or 'yawn'. Pressing 'y' labels the frame as a yawn, and 'n' labels it as normal. These labels are saved to `data/labels.csv`.
2.  **Frame Extraction:** The `src/extract_images.py` script reads the `labels.csv` file and extracts the corresponding frames from the source videos, saving them into categorized folders (`normal`/`yawn`).
3.  **Face Cropping:** `src/crop_faces.py` processes the extracted frames, uses MediaPipe's Face Detection to locate the face in each image, and saves the cropped face for training.
4.  **Preprocessing & Splitting:** The `notebooks/preprocess_faces.ipynb` notebook resizes the cropped faces to a consistent size (e.g., 64x64 or 224x224), converts them to the required color format, and splits the dataset into training, validation, and test sets.
5.  **Model Training:** Several models were trained and experimented with, including a custom CNN and multiple variations of a fine-tuned MobileNetV2. The training process, including data augmentation and performance evaluation, is documented in the Jupyter notebooks (`notebooks/`). Experiment tracking was managed using Weights & Biases.
6.  **Real-time Detection:** The final application, `src/realtime_drowsiness.py`, captures video from a webcam, uses MediaPipe to detect and track the driver's face in real-time, and feeds the cropped face into the trained model (`mobilenet4`) to classify the driver's state. An audible beep is triggered if a yawn is detected.

## Project Structure

```
├── data/              # Dataset files, managed by DVC
│   ├── labels.csv     # Manually created labels for video frames
│   ├── raw.dvc        # Raw video data
│   ├── processed.dvc  # Cropped face images
│   └── ...            # Train/val/test splits
├── models/            # Trained model files, managed by DVC
│   ├── mobilenet4/    # The final model used for real-time detection
│   └── ...            # Other experimental models
├── notebooks/         # Jupyter notebooks for preprocessing and training
│   ├── preprocess_faces.ipynb
│   ├── train_cnn_v1.ipynb
│   └── train_mobilenet*.ipynb
├── src/               # Source code
│   ├── crop_faces.py             # Script to detect and crop faces from frames
│   ├── extract_images.py         # Script to extract frames from videos based on labels
│   ├── realtime_drowsiness.py    # Main script for real-time detection
│   └── video_labeling.py         # Tool for labeling video data
├── requirements.txt   # Python dependencies
└── README.md
```

## Getting Started

### Prerequisites

*   Python 3.8+
*   [DVC](https://dvc.org/doc/install)
*   A webcam for real-time detection

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Omar-Hariri/driver-drowsiness-detection.git
    cd driver-drowsiness-detection
    ```

2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Pull the data and models using DVC:**
    This will download the dataset and the trained models managed by DVC.
    ```bash
    dvc pull
    ```

## Usage

### Real-Time Drowsiness Detection

To run the real-time detection system using your webcam, execute the following command from the root directory of the project:

```bash
python src/realtime_drowsiness.py
```

The application will open a window showing your webcam feed. It will draw a bounding box around your face and display the predicted status ("NORMAL" or "YAWN"). A beep will sound if a yawn is detected. Press 'q' to quit.

### Training Your Own Model

The entire MLOps pipeline, from data processing to training, is reproducible.

1.  **Prepare Data:** Add your own videos to the `data/raw` directory.
2.  **Label Videos:** Use the `src/video_labeling.py` script to create labels for your new videos.
3.  **Process Data:** Run the data extraction (`src/extract_images.py`) and face cropping (`src/crop_faces.py`) scripts.
4.  **Preprocess and Split:** Run the `notebooks/preprocess_faces.ipynb` notebook to prepare the final train/validation/test sets.
5.  **Train:** Use one of the training notebooks (e.g., `notebooks/train_mobilenet4.ipynb`) to train a new model on the updated dataset.
