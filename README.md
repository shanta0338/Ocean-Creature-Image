# 🌊 Ocean Creature Image Classification

Deep learning–based image classification system for identifying ocean creatures from images using a trained convolutional neural network.

**Live Model (HuggingFace):**
[https://huggingface.co/Shanta1875/Ocean_Creature_Image_Classification](https://huggingface.co/Shanta1875/Ocean_Creature_Image_Classification)

---

# 📌 Project Overview

This project focuses on **multi-class image classification of ocean creatures** using deep learning and transfer learning techniques.
The model is trained to accurately recognize different marine species from images and can be used in:

* Marine biodiversity monitoring
* Educational tools
* Ocean research automation
* AI-based wildlife detection systems

---

# 🧠 Model Details

| Attribute  | Value                                   |
| ---------- | --------------------------------------- |
| Model Type | Deep Learning (CNN / Transfer Learning) |
| Framework  | PyTorch                                 |
| Task       | Multi-class Image Classification        |
| Domain     | Marine / Ocean Creatures                |
| Model Size | ~80 MB                                  |
| Deployment | HuggingFace Model Hub                   |

---

# 🐠 Supported Classes (Example)

The model is trained to classify multiple ocean creatures such as:

* Shark
* Dolphin
* Whale
* Octopus
* Jellyfish
* Sea turtle
* Crab
* Starfish

*(Actual classes depend on training dataset)*

---

# 📂 Model Access

You can directly download or load the trained model from HuggingFace:

**Model Repository:**
[https://huggingface.co/Shanta1875/Ocean_Creature_Image_Classification](https://huggingface.co/Shanta1875/Ocean_Creature_Image_Classification)

---

# 🚀 How to Use the Model

## 1. Install dependencies

```bash
pip install torch torchvision pillow huggingface_hub
```

---

## 2. Load model from HuggingFace

```python
from huggingface_hub import hf_hub_download
import torch

# download model
model_path = hf_hub_download(
    repo_id="Shanta1875/Ocean_Creature_Image_Classification",
    filename="model.pth"
)

# load model
model = torch.load(model_path, map_location="cpu")
model.eval()
```

---

## 3. Run prediction on image

```python
from PIL import Image
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

img = Image.open("test.jpg").convert("RGB")
img = transform(img).unsqueeze(0)

with torch.no_grad():
    output = model(img)
    pred = output.argmax(1)

print("Predicted class:", pred.item())
```

---
