# 🩺 Chest X-Ray Pneumonia Detection using Deep Learning

This project aims to detect pneumonia from chest X-ray images using a Convolutional Neural Network (CNN) built with TensorFlow/Keras. It classifies images as either **Normal** or **Pneumonia**, assisting in faster and more accurate diagnoses.

---

## 👨‍💻 Team Members

- **Abhinash Pradhan** - 210301120114  
- **Preeti Ranjan Sarangi** - 210301120123  
- **Swapnil Das** - 210301120126

---

## 🧠 Project Overview

Chest X-rays are a crucial diagnostic tool for pneumonia, but manual diagnosis can be slow and prone to error. In this project, we:

- Built a CNN to classify chest X-rays
- Trained the model on publicly available medical data
- Evaluated its performance with various metrics
- Visualized predictions and deployed a testing pipeline

---

## 📂 Project Structure

```
📁 chest_xray/
├── chest_xRay_pneumonia.ipynb       # Main notebook
├── README.md                        # Project description
├── requirements.txt                 # Dependencies
└── chest_xray_pneumonia_model.h5    # (Not uploaded due to GitHub file size limit)
```

---

## 📦 Dataset

We used the **Chest X-Ray Images (Pneumonia)** dataset from [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).  
It includes:

- `train/` — for model training  
- `test/` — for final evaluation  
- `val/` — optional, for validation

---

## 🧪 Model Summary

- **Input Shape**: (220, 220, 3)  
- **Architecture**: Custom CNN with Conv2D, MaxPooling, Dropout, Flatten, and Dense layers  
- **Output**: Binary classification (`Normal` or `Pneumonia`)  
- **Loss Function**: Binary Crossentropy  
- **Optimizer**: Adam

---

## 🔍 Evaluation Metrics

- Accuracy  
- Precision / Recall / F1-score  
- Confusion Matrix  
- ROC-AUC Curve  

The model achieved **over 90% validation accuracy** during training.  
Note: Accuracy may vary depending on training epochs, augmentation, and batch size.

---

## 💻 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/kirtivardhan80/chest_xray_pneumonia.git
cd chest_xray_pneumonia
```

### 2. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 3. Run the Jupyter Notebook

```bash
jupyter notebook chest_xRay_pneumonia.ipynb
```

### 4. Train the Model or Load a Pretrained One

To save your model after training:

```python
model.save("chest_xray_pneumonia_model.h5")
```

---

## 🚫 Model File Note

Due to GitHub's 100MB file limit, the trained model `.h5` is **not included** here.

If you'd like to test without retraining, download the model from Google Drive:

📥 [Download chest_xray_pneumonia_model.h5](https://drive.google.com/file/d/YOUR_MODEL_ID/view?usp=sharing)

After downloading, place the file inside the project folder.

---

## 📸 Sample Predictions

| Input X-ray                | Predicted Label |
|---------------------------|-----------------|
| ![Normal](images/sample1.jpg)    | Normal          |
| ![Pneumonia](images/sample2.jpg) | Pneumonia       |

> Add these images inside an `images/` folder if you want them to render.

---

## 📦 Requirements

```txt
tensorflow
keras
numpy
opencv-python
matplotlib
pandas
scikit-learn
jupyter
```

To install:

```bash
pip install -r requirements.txt
```

---

## 🚀 Future Enhancements

- 🔍 Use pretrained models like MobileNetV2 or ResNet for better accuracy  
- 🌈 Add Grad-CAM or heatmaps for model explainability  
- 🌐 Deploy the model with Streamlit for easy web interface  
- 🧪 Extend dataset with more samples for generalization  
- 📱 Optimize for mobile with TensorFlow Lite  

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

You are free to use, modify, and distribute this code for personal, academic, or commercial purposes, as long as proper credit is given.

> ⚠️ Note: This project is for **educational and research purposes only**. It is **not intended for clinical or diagnostic use** without validation from medical professionals.

---

## 🙏 Acknowledgments

- [Kaggle Chest X-Ray Dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)  
- TensorFlow / Keras documentation  
- OpenCV, Matplotlib, NumPy communities

---
