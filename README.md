---

```markdown
# 🧠 ResNet-50 + YOLOv3 Object Detection  
> Internship Assignment – Full Object Detection Pipeline on Pascal VOC

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/framework-PyTorch-red)
![Status](https://img.shields.io/badge/status-Prototype-yellow)

---

## 🔍 Project Overview

This project implements an object detection pipeline using a custom YOLOv3-style detector with a **ResNet-50** backbone. It is trained and evaluated on the **Pascal VOC 2007** dataset.

🎯 Developed as part of an AI internship assignment:
- ✅ End-to-end training pipeline  
- ✅ YOLO-style head and anchor setup  
- ✅ Inference with GT/prediction overlays  
- ✅ COCO-style mAP evaluation  
- ✅ Results saved in structured outputs  
- ✅ Documentation included  

---

## 🗂️ Project Structure

```

resnet50-yolov3-detector/
│
├── data/                    ← Pascal VOC dataset
├── models/                  ← Trained checkpoints (.pth)
├── notebooks/
│   ├── train.ipynb          ← Model training
│   └── inference.ipynb      ← Evaluation + visualization
├── outputs/
│   ├── inference/           ← Output PNGs (GT & predictions)
│   └── metrics.csv          ← Evaluation scores (COCO mAP)
├── report/
│   └── assignment-report.pdf← Internship report document
├── requirements.txt         ← Python dependencies
├── README.md
└── .gitignore

````

---

## 🚀 Setup & Run

### 🔧 Install Requirements

```bash
pip install -r requirements.txt
````

Or launch the notebooks directly in [Google Colab](https://colab.research.google.com/).

---

### 📦 Dataset

The Pascal VOC 2007 dataset is auto-downloaded via PyTorch:

```python
from torchvision.datasets import VOCDetection
VOCDetection(root='data/VOCdevkit', year='2007', image_set='trainval', download=True)
VOCDetection(root='data/VOCdevkit', year='2007', image_set='test', download=True)
```

---

## 🏋️‍♂️ Model Training

Run `notebooks/train.ipynb`:

* Loads ResNet-50 backbone
* Adds YOLOv3 detection head
* Trains on VOC 2007 `trainval` set
* Saves `.pth` model checkpoints under `models/`

Training parameters are configured in `Config` (e.g., `img_size`, `batch_size`, `epochs=12`).

---

## 🎯 Inference & Evaluation

Run `notebooks/inference.ipynb`:

* Loads saved checkpoint
* Runs inference on VOC test images
* Saves overlaid images to `outputs/inference/`
* Evaluates with COCO API (AP, AR)
* Saves scores to `outputs/metrics.csv`

---

## 📊 Results Preview

<table>
  <tr>
    <td><img src="outputs/inference/10.png" width="300"/></td>
    <td><img src="outputs/inference/96.png" width="300"/></td>
  </tr>
  <tr>
    <td align="center">Green = Ground Truth</td>
    <td align="center">Red = YOLOv3 Predictions</td>
  </tr>
</table>

This prototype model was trained for **12 epochs** on Pascal VOC using a custom YOLOv3 head and ResNet-50 backbone.

✅ It demonstrates a fully functional object detection pipeline:

* Prediction, decoding, and visualization
* Ground truth overlays for visual inspection
* mAP evaluation using COCO API

📈 Further training (e.g., 50+ epochs with better anchor handling) is expected to significantly improve performance.

---

## 📄 Documentation

* 📘 **Assignment Report**: [`report/assignment-report.pdf`](report/assignment-report.pdf)
* 📦 **Requirements File**: [`requirements.txt`](requirements.txt)
* ✅ In-notebook markdown cells explain each step clearly

---

## ✍️ Author

**Tejash Pandey**
GitHub: [@tejash-300](https://github.com/tejash-300)
AI Internship — 2025

---

## 🔧 Future Work

* [ ] Add anchor-to-GT assignment logic
* [ ] Switch to CIoU/GIoU loss
* [ ] Train for 50+ epochs
* [ ] Add video or GIF demo from inference images

```

---


```

