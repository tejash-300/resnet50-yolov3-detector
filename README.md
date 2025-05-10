

```markdown
# ResNet-50 + YOLOv3 Object Detection

This repository contains a PyTorch implementation of a YOLOv3-style object detector built on top of a ResNet-50 backbone, trained and evaluated on the Pascal VOC 2007 dataset. It includes:

- End-to-end training pipeline  
- Inference & visualization of predictions vs. ground truth  
- COCO-style evaluation (mAP)  
- All code, model weights, data, and outputs organized for easy reproduction  

---

## 📁 Repository Structure

```

resnet50-yolov3-detector/
├── data/                   # Pascal VOC dataset (2007 trainval + test)
├── models/                 # Saved checkpoints (yolo\_epoch\*.pth)
├── notebooks/              # Colab notebooks for training & inference
│   ├── train.ipynb
│   └── inference.ipynb
├── outputs/
│   ├── inference/          # Overlay PNGs of GT (green) vs. preds (red)
│   └── metrics.csv         # COCO-style AP/AR results
├── code/                   # (Optional) Python modules (model, loss, utils)
├── .gitignore              # Ignored files/folders
└── .gitattributes          # Git LFS config for large files

````

---

## 🚀 Quickstart

### 1. Clone the repo & install prerequisites

```bash
git clone https://github.com/tejash-300/resnet50-yolov3-detector.git
cd resnet50-yolov3-detector

# (optional) create a conda environment
conda create -n yolo-env python=3.8 -y
conda activate yolo-env

# install Python packages
pip install torch torchvision pycocotools matplotlib tqdm
````

### 2. Download the Pascal VOC 2007 data

In a Colab cell or terminal:

```bash
# inside a notebook or script you have
from torchvision.datasets import VOCDetection
VOCDetection(root='data/VOCdevkit', year='2007', image_set='trainval', download=True)
VOCDetection(root='data/VOCdevkit', year='2007', image_set='test',    download=True)
```

This will populate `data/VOCdevkit/`.

### 3. Train the model

Open **`notebooks/train.ipynb`** in Google Colab:

1. Mount your Google Drive if you like, or work in `/content/` directly.
2. Run the cells in order:

   * **Cell 1–2**: setup folders & imports
   * **Cell 3**: hyperparameters (`cfg.epochs`, `cfg.img_size`, etc.)
   * **Cell 4–6**: data loaders
   * **Cell 7**: YOLO loss
   * **Cell 8–9**: training loop (SGD + scheduler)
3. Checkpoints (`models/yolo_epoch{n}.pth`) will be saved each epoch.

### 4. Inference & visualization

Open **`notebooks/inference.ipynb`**:

1. **Cell X**: load your best checkpoint.
2. **Cell Y**: run inference on VOC test set (or first 100 images).
3. **Cell Z**: save overlay PNGs to `outputs/inference/`.

You’ll see green boxes for ground truth and red boxes for your model’s predictions.

### 5. Evaluate (mAP)

In **`notebooks/inference.ipynb`** or **`Cell Z`**:

1. Fake a COCO-style GT from VOC annotations.
2. Decode your model’s raw outputs into COCO format.
3. Run `pycocotools.COCoeval` to compute AP@\[.5:.95], AP50, AR, etc.
4. Results are saved to `outputs/metrics.csv`.

---

## 💡 Tips & Next Steps

* **Improve your loss**: swap MSE for CIoU/GIoU and add anchor-to-GT matching to boost performance.
* **Train longer**: 30–50 epochs and unfreeze the ResNet backbone for fine-tuning.
* **Data augmentation**: random flips, color jitter, and mosaic aug can help.
* **Experience report**: don’t forget to add your AI-tool prompt history and reflections in `/report/experience_report.pdf` or `.md`.

---

## 📜 License & Acknowledgements

This project was developed as part of an AI internship assignment. The YOLOv3-style head is inspired by the official YOLOv3 paper by Redmon et al. and the Ultralytics implementation.

Feel free to adapt and extend for your own research or coursework!
