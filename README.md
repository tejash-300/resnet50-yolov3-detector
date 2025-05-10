



```markdown
# 🧠 ResNet-50 + YOLOv3 Object Detection  
> **AI Internship Assignment — Object Detection Pipeline with ResNet + YOLOv3**

![Python](https://img.shields.io/badge/python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/framework-PyTorch-red)
![Status](https://img.shields.io/badge/status-Prototype-yellow)

---

## 🔍 Overview

This repository implements a YOLOv3-style object detector using a **ResNet-50** backbone.  
It is trained and evaluated on the **Pascal VOC 2007** dataset.

✅ Key Features:

- End-to-end training and evaluation pipeline  
- Custom YOLO-style head with anchors  
- Inference with ground truth + prediction overlays  
- COCO-style mAP evaluation  
- Runs in Colab or locally with PyTorch  
- Includes full report & requirements  

---

## 📁 Project Structure

```

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

## 🚀 Getting Started

<details>
<summary><strong>📦 Setup & Install</strong></summary>

```bash
# Clone repo
git clone https://github.com/tejash-300/resnet50-yolov3-detector.git
cd resnet50-yolov3-detector

# Install dependencies
pip install -r requirements.txt
````

</details>

<details>
<summary><strong>📥 Download Pascal VOC Dataset</strong></summary>

Dataset is auto-downloaded in the notebook using torchvision:

```python
from torchvision.datasets import VOCDetection

# Train/val split
VOCDetection(root='data/VOCdevkit', year='2007', image_set='trainval', download=True)

# Test split
VOCDetection(root='data/VOCdevkit', year='2007', image_set='test', download=True)
```

</details>

---

## 🏋️‍♂️ Model Training

Open `notebooks/train.ipynb` and run:

* Initializes ResNet-50 as backbone
* Adds YOLOv3 multi-scale detection head
* Uses SGD optimizer with StepLR
* Trains for `12` epochs
* Saves `.pth` checkpoints in `models/`

📌 You can modify `Config` to adjust epochs, image size, learning rate, etc.

---

## 🧪 Inference & Evaluation

Open `notebooks/inference.ipynb`:

* Loads trained model checkpoint
* Runs inference on VOC test images
* Overlays red prediction boxes & green GT boxes
* Saves visual outputs to `outputs/inference/`
* Evaluates using `pycocotools` (COCO API)
* mAP results are saved to `outputs/metrics.csv`

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

This prototype model was trained for **12 epochs** using Pascal VOC data.

✔️ The system demonstrates the full detection workflow:

* Input → Prediction → Decoding → Overlay → Evaluation
* Ground truth is used for validation and visualization
* Results show a functioning pipeline under limited compute

📌 Future training (e.g. 50+ epochs, CIoU loss, anchor assignment) will improve mAP.

---

## 📄 Documentation

* 📘 **Assignment Report**: [`report/assignment-report.pdf`](report/assignment-report.pdf)
* 📦 **Dependencies**: [`requirements.txt`](requirements.txt)
* 🧠 **Code Walkthrough**: Inside Colab notebooks via markdown cells

---

## ✍️ Author

**Tejash Pandey**
GitHub: [@tejash-300](https://github.com/tejash-300)
*This project was developed as part of an AI internship assignment in 2025.*

---

## 🔧 Future Enhancements

* [ ] Anchor-to-GT assignment (for better supervision)
* [ ] Switch to CIoU/GIoU loss for improved localization
* [ ] Train for 50+ epochs and unfreeze ResNet backbone
* [ ] Add video/GIF-based demo from inference PNGs
* [ ] Upload model demo on Hugging Face or Gradio

```

---


```
