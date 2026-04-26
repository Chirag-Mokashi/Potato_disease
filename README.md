# Potato Disease Classification using Deep Learning

A convolutional neural network (CNN) trained on the PlantVillage dataset to classify potato leaf images into three disease categories.

## Overview

Potato crops are highly susceptible to fungal diseases that can devastate yields. This project trains a deep learning model to identify disease from leaf images, enabling early detection that can guide timely treatment.

**Classified conditions:**
- **Early Blight** — caused by *Alternaria solani*, appears as dark concentric lesions
- **Late Blight** — caused by *Phytophthora infestans*, appears as water-soaked lesions
- **Healthy** — no disease present

## Dataset

**PlantVillage** — publicly available agricultural image dataset

| Class | Description |
|-------|-------------|
| `Potato___Early_blight` | Fungal early blight infection |
| `Potato___Late_blight` | Oomycete late blight infection |
| `Potato___healthy` | Healthy potato leaf |

Images are stored under `Training/VillagePlant/PlantVillage/` organised by class folder.

## Model

A CNN trained using TensorFlow/Keras:

- Input: RGB leaf images (resized)
- Architecture: Convolutional + pooling layers with dense classification head
- Output: Softmax over 3 classes
- Saved model: `potatoes.h5`

## Repository Structure

```
Potato_disease/
    Training/
        Training.ipynb      Model training notebook
        VillagePlant/
            PlantVillage/
                Potato___Early_blight/
                Potato___Late_blight/
                Potato___healthy/
    saved_models/           Versioned saved model exports
    potatoes.h5             Trained Keras model
    requirements.txt
    setup.py
```

## Setup

```bash
git clone https://github.com/Chirag-Mokashi/Potato_disease
cd Potato_disease
pip install -r requirements.txt
```

## Training

Open `Training/Training.ipynb` and run all cells. The notebook:
1. Loads and preprocesses PlantVillage images
2. Splits into train / validation / test sets
3. Trains the CNN
4. Evaluates accuracy on the test set
5. Saves the model to `potatoes.h5` and `saved_models/`

## Inference

```python
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("potatoes.h5")
class_names = ["Early Blight", "Late Blight", "Healthy"]

img = Image.open("leaf.jpg").resize((256, 256))
img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
pred = model.predict(img_array)
print(class_names[np.argmax(pred)])
```

## Tech Stack

- Python, TensorFlow / Keras
- NumPy, Matplotlib
- PlantVillage dataset