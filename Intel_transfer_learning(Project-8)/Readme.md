

#  Day 19 — Transfer Learning with ResNet50 on Intel Image Classification Dataset

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3.0-red.svg?style=flat-square)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg?style=flat-square)](LICENSE)

This project demonstrates **Transfer Learning** using a pre-trained **ResNet50** model on the **Intel Image Classification Dataset**.  
It’s part of the *90 Days of Machine Learning Challenge* by **Yash Bishnoi**.

---

## Dataset Overview

The dataset used is the **Intel Image Classification Dataset**, which contains natural scene images categorized into **6 classes**:

-  buildings  
-  forest  
-  glacier  
-  mountain  
-  sea  
-  street

###  Folder Structure

Intel_transfer_learning/
├── seg_train/
│   ├── buildings/
│   ├── forest/
│   ├── glacier/
│   ├── mountain/
│   ├── sea/
│   └── street/
└── seg_test/
├── buildings/
├── forest/
├── glacier/
├── mountain/
├── sea/
└── street/

Each folder contains JPEG images of natural scenes belonging to that category.

---

##  Setup Instructions

### 1️ Create Virtual Environment
```bash
python -m venv venv

2️ Activate Environment
	•	macOS/Linux:

source venv/bin/activate

	•	Windows (PowerShell):

venv\Scripts\activate

3️ Install Dependencies

pip install -r requirements.txt

4️ Dataset Placement

Place the seg_train and seg_test folders inside:

Intel_transfer_learning/


⸻

 Requirements

torch==2.3.0
torchvision==0.18.0
torchaudio==2.3.0
Pillow==10.3.0
numpy==1.26.4
matplotlib==3.9.0
tqdm==4.66.4
kaggle==1.6.6
torchsummary==1.5.1


⸻

 Run Training

python resnet50_intel_transfer.py

Expected Output:

 Classes: ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
 Training samples: 14034, Validation samples: 3000
 Using device: mps


⸻

 Model Architecture Diagram

The ResNet50 model can be visualized using torchsummary:

from torchvision.models import resnet50
from torchsummary import summary
import torch

device = torch.device("mps" if torch.has_mps else "cpu")
model = resnet50(pretrained=True).to(device)
summary(model, input_size=(3, 224, 224))

This prints a layer-by-layer summary including output shapes and number of parameters.

⸻

 Model Checkpoint

After training completes, the best model is saved automatically as:

resnet50_intel_best.pth

It can be loaded later for inference using the predict_image() function in the script.

⸻

 Example Inference

from PIL import Image
from torchvision import transforms
import torch

# Load model (after training)
model.load_state_dict(torch.load("resnet50_intel_best.pth"))
model.eval()

# Run prediction
predict_image(model, "Intel_transfer_learning/seg_test/mountain/12345.jpg")

Output:

 Image: 12345.jpg →  Predicted class: mountain


⸻

 Day 19 — Results & Observations

Model: Pre-trained ResNet50
Dataset: Intel Image Classification (6 classes)
Device: macOS MPS
Training: 10 epochs

Epoch	Train Loss	Train Acc	Val Loss	Val Acc
1	0.5680	0.7894	0.2745	0.9017
2	0.4681	0.8226	0.2429	0.9113
3	0.4295	0.8400	0.2607	0.8977
4	0.4316	0.8355	0.2483	0.9053
5	0.4246	0.8383	0.2217	0.9147
6	0.4063	0.8400	0.2216	0.9193
7	0.4053	0.8478	0.2414	0.9107
8	0.4045	0.8445	0.2285	0.9137
9	0.3980	0.8516	0.2236	0.9160
10	0.3920	0.8524	0.2157	0.9197

 Training Time: 26 minutes 56 seconds
 Best Validation Accuracy: 91.97%

 Observations
	1.	Rapid Convergence:
	•	Achieved >90% validation accuracy by Epoch 1–2.
	2.	Loss Trends:
	•	Training loss steadily decreased.
	•	Validation loss slightly fluctuated but decreased overall.
	3.	Accuracy Trends:
	•	Validation accuracy peaked at 91.97%, showing strong generalization.
	4.	Effectiveness of Transfer Learning:
	•	Freezing most layers while retraining the final layer allowed fast learning with high accuracy.
	5.	macOS MPS Backend Performance:
	•	Training ran efficiently on Apple Silicon (~27 mins for 10 epochs).
	•	Known MPS fixes (num_workers=0, .float() for accuracy) were applied.

 Conclusion:
	•	Transfer learning with ResNet50 works very well for Intel Image Classification.
	•	Day 20 can focus on fine-tuning deeper layers to improve accuracy further.

⸻

 Notes for macOS Users (MPS Backend)

 The code automatically detects and uses:

device = "mps"

 Known Fixes:
	•	Set num_workers=0 in DataLoader
	•	Replace .double() with .float() when computing accuracy

⸻

 Key Concepts Covered
	•	Transfer Learning
	•	Fine-tuning Pre-trained CNNs
	•	Data Augmentation & Normalization
	•	Model Evaluation
	•	macOS MPS GPU Acceleration

⸻

 Future Work (Day 20 Preview)

 Fine-Tune deeper ResNet layers
 Add Learning Rate Scheduler
 Implement Early Stopping
 Visualize accuracy/loss curves with TensorBoard

⸻

 Author

Yash Bishnoi
 MSc Computer Science | University of Dundee
 Day 19 — 90 Days of Machine Learning Challenge
 Passionate about Machine Learning, Deep Learning & AI

