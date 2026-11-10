# Real-Time Face Mask Detection with YOLO26

## Project Overview

This project is a real-time computer vision system that detects whether people are wearing a face mask correctly, incorrectly, or not at all. A custom **YOLO26 object detection** model was trained on a public face mask dataset using **Roboflow**, then deployed on a live webcam and video using **Ultralytics** and **OpenCV**.

Built as the final project for the **SDAIA Academy – Computer Vision for Developers** program.

## Problem Description

Wearing face masks correctly is important in hospitals, clinics, labs, and crowded public places, but checking this manually is slow and inconsistent.

**Purpose:** automatically detect faces in a camera feed and classify each one by mask status, so mask compliance can be monitored without human effort.

**Expected output:** a live video stream where every face is surrounded by a bounding box labeled with one of three classes:

| Class | Meaning |
|---|---|
| `with_mask` | Mask worn correctly |
| `without_mask` | No mask |
| `mask_weared_incorrect` | Mask worn incorrectly (e.g., nose uncovered) |

## Dataset & Model Used

**Dataset:** [Face Mask Detection – Kaggle (andrewmvd)](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)
- 848 images uploaded to Roboflow, all pre-annotated with bounding boxes (PASCAL VOC format)
- 3 classes: `with_mask`, `without_mask`, `mask_weared_incorrect`
- Split: **70% train (594)** / **20% validation (169)** / **10% test (85)**
- Preprocessing: Roboflow's default preprocessing steps applied when generating the dataset version

**Model:** **YOLO26 Object Detection (Nano)** by Ultralytics
- Computer vision task: **object detection**
- Chosen because it is fast on CPU, NMS-free, and small enough to run in real time on a laptop
- Trained on Roboflow using **transfer learning** from pretrained weights
- Training was early-stopped once validation mAP stopped improving

## Workflow / Architecture

```
Kaggle Dataset
     │
     ▼
Roboflow: upload images + annotations ──► create dataset version (train / valid / test)
     │
     ▼
Train YOLO26 Nano on Roboflow ──► download trained weights (weights.pt)
     │
     ▼
Webcam / Video ──► Frames (OpenCV) ──► YOLO26 ──► Predictions ──► Display with boxes & labels
```

## Results & Evaluation

| Metric | Score |
|---|---|
| **mAP@50 (Test set)** | **76.0%** |
| mAP@50 (Validation set) | 68.2% |
| Precision (Validation) | 83.1% |
| Recall (Validation) | 61.1% |
| F1 Score (Validation) | 70.4% |
| Inference latency | ~6.3 ms |

Precision is high (when the model predicts a class, it is usually right), while recall is lower, meaning some faces — especially small or far-away ones — are missed.

**Demo on an unseen video:**




https://github.com/user-attachments/assets/ecc274f5-28de-4a0f-95fb-4c653026a99c




**Successful prediction:** most masked faces in a crowded scene are correctly detected as `with_mask`, and an unmasked person is detected as `without_mask`.

**Failure case:** some faces are missed, such as a man without a mask in the middle of the crowd and people wearing patterned cloth masks. The dataset contains mostly medical masks, so the model struggles with different mask styles and far-away faces.

## Technologies Used

- Python
- Ultralytics (YOLO26)
- OpenCV
- Roboflow (dataset management + training + workflows)
- Kaggle (dataset source)
- Jupyter Notebook / VS Code
- GitHub

## How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/Danah-99/sdaia-academy-project.git
   cd sdaia-academy-project
   ```

2. Install the requirements:
   ```bash
   pip install -U ultralytics opencv-python
   ```

3. Open the notebook `Real_Time_YOLO_Webcam_Object_Detection_Roboflow_Lab.ipynb`, or run this code:
   ```python
   from ultralytics import YOLO
   import cv2

   model = YOLO("weights.pt")
   cap = cv2.VideoCapture(0)  # 0 = webcam, or put a video file path

   while True:
       ret, frame = cap.read()
       if not ret:
           break
       results = model(frame, conf=0.25)
       cv2.imshow("Face Mask Detector", results[0].plot())
       if cv2.waitKey(1) & 0xFF == ord("q"):
           break

   cap.release()
   cv2.destroyAllWindows()
   ```

4. Press **q** to close the window.

**Real application:** the model runs in real time on a laptop CPU, so it could be used at a hospital entrance or an access-control gate to monitor mask use. For faster deployment, it can be exported to ONNX:
```python
YOLO("weights.pt").export(format="onnx")
```

## Future Improvements

- Collect more images for `mask_weared_incorrect`, which has far fewer examples than the other classes.
- Add more images of cloth and patterned masks, and small/distant faces.
- Add data augmentation (flip, brightness, blur, rotation) to handle different lighting and angles.
- Train longer or use a larger model (YOLO26 Small/Medium) to improve recall.
- Deploy as a web or mobile app, or apply quantization for edge devices.

## SDAIA Academy GitHub Repository Link

Danah Alanazi & Norah Alhazani @norah-21

[SDAIA Academy GitHub Repository](https://github.com/SDAIAAcademy)
