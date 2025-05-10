

---

````markdown
# 🧠 ResNet-50 + YOLOv3 Object Detection  
> Internship Assignment – Full Object Detection Pipeline on Pascal VOC

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/framework-PyTorch-red)
![Status](https://img.shields.io/badge/status-Prototype-yellow)

---

## 🔍 Project Overview  
This project implements an end-to-end object detection pipeline using a YOLOv3-style head on top of a ResNet-50 backbone. It is trained and evaluated on the Pascal VOC 2007 dataset.

🛠️ Built as part of an AI internship assignment with the following key features:

- ✔️ Custom YOLO-style model with ResNet-50 backbone
- ✔️ Dataset loading, anchor setup, and loss function
- ✔️ End-to-end training in PyTorch
- ✔️ Inference and image overlay
- ✔️ COCO-style mAP evaluation
- ✔️ Visual and quantitative outputs saved
- ✔️ Ready-to-use Google Colab notebooks

---

## 📁 Project Structure

```text
resnet50-yolov3-detector/
├── data/                    ← Pascal VOC dataset
├── models/                  ← Trained model checkpoints (.pth)
├── notebooks/
│   ├── train.ipynb          ← Model training
│   └── inference.ipynb      ← Evaluation + visualization
├── outputs/
│   ├── inference/           ← Output PNGs (GT & predictions)
│   └── metrics.csv          ← Evaluation scores (COCO mAP)
├── report/
│   └── assignment-report.pdf← Final submission document
├── requirements.txt         ← Python dependencies
└── README.md
````

---

## 🚀 Setup & Run

### 🔧 Install Requirements

```bash
pip install -r requirements.txt
```

Alternatively, open the notebooks in **Google Colab** for hassle-free execution.

---

### 📦 Dataset

The Pascal VOC 2007 dataset is automatically downloaded:

```python
from torchvision.datasets import VOCDetection

VOCDetection(root='data/VOCdevkit', year='2007', image_set='trainval', download=True)
VOCDetection(root='data/VOCdevkit', year='2007', image_set='test', download=True)
```

---

## 🏋️ Model Training

Open and run `notebooks/train.ipynb`:

* Uses a pretrained ResNet-50 backbone
* Adds 3-scale YOLOv3 detection heads
* Trains on VOC 2007 `trainval` set
* Saves `.pth` model checkpoints under `/models`

> 🔁 Trained for `12 epochs` as a prototype. Can be extended to 50+ epochs.

---

## 🎯 Inference & Evaluation

Open `notebooks/inference.ipynb`:

* Loads trained model checkpoint
* Performs object detection on VOC test images
* Saves images with:

  * 🟩 Green boxes: Ground truth
  * 🟥 Red boxes: Predictions
* Calculates **COCO-style mAP**
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

> ⚠️ This is a **prototype model**, trained for only 12 epochs without anchor tuning or advanced loss. It demonstrates the full functionality of the detection pipeline. Future training (e.g., 50+ epochs) is expected to yield better metrics.

---

## 📄 Documentation

* 📘 Assignment Report: `report/assignment-report.pdf`
* 📦 Requirements File: `requirements.txt`
* ✅ In-notebook markdown blocks explain code logic and flow

---

## ✍️ Author

**Tejash Pandey**
GitHub: [@tejash-300](https://github.com/tejash-300)
AI Internship – 2025

---

## 🔧 Future Work

* ✅ Add anchor-to-GT matching (better target assignment)
* ✅ Switch to CIoU or GIoU loss
* ✅ Train for 50+ epochs with augmented data
* ✅ Export to ONNX / TorchScript for deployment
* ✅ Add live webcam/video inference support

---

> ⭐ *Feel free to fork, explore, or extend this project.*

```



