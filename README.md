# 🦷 Oral Disease Classification Using Deep Learning  

> **Detect Oral Diseases with AI!** An advanced deep learning model for classifying various oral diseases using medical imaging.

---

## 🌟 Highlights  

- 🚀 **MobileNetV2 Backbone**: Built on a pre-trained MobileNetV2 for robust feature extraction.  
- 📊 **High Performance**: Achieves significant accuracy for multi-class classification.  
- 🧪 **Comprehensive Analysis**: Includes confusion matrices, training/validation graphs, and classification reports.  
- 🔄 **Automated Workflow**: Data preprocessing, model training, and evaluation in one streamlined script.  

---

## 📜 Table of Contents  

1. [Project Overview](#-project-overview)  
2. [Features](#-features)  
3. [Dataset](#-dataset)  
4. [Installation](#-installation)  
5. [Model Architecture](#-model-architecture)  
6. [Results](#-results)  
7. [How to Use](#-how-to-use)  
8. [Future Enhancements](#-future-enhancements)  
9. [Acknowledgments](#-acknowledgments)  
10. [Contributors](#-contributors)  

---

## 🧐 Project Overview  

This project is designed to assist dental practitioners and researchers by automating the detection and classification of oral diseases using image data. With categories like **healthy gums**, **cavities**, and more, this tool brings AI into the world of dental care!  

💡 **Goal**: Classify oral disease images into categories like:  
1. Healthy Gums 🦷  
2. Cavities 🛠️  
3. Gum Disease 🌿  
4. ...and more!  

---

## ✨ Features  

- 📈 **Multi-Class Classification**: Handles multiple oral disease types with categorical labels.  
- 🧹 **Data Preprocessing**: Automatic resizing, normalization, and one-hot encoding.  
- 🔧 **Model Customization**: Modifiable architecture with tunable hyperparameters.  
- 📊 **Visual Analytics**: Training/validation graphs, confusion matrices, and detailed classification reports.  
- 💾 **Model Saving**: Trained models are saved in `.keras` format for future use.  

---

## 📁 Dataset  

The dataset consists of categorized images stored in:  
- **Training Directory**: `/Dataset/TRAIN`  
- **Testing Directory**: `/Dataset/TEST`  

Each category contains labeled images, automatically loaded and preprocessed during training.  

Example structure:  
```plaintext
Dataset/  
└── TRAIN/  
    ├── Healthy_Gums/  
    ├── Cavities/  
    └── Gum_Disease/  
└── TEST/  
    ├── Healthy_Gums/  
    ├── Cavities/  
    └── Gum_Disease/  
```  

Images are resized to **224x224** and normalized to [0,1]. Labels are one-hot encoded for classification.  

---

## 🧠 Model Architecture  

This model leverages the power of **MobileNetV2**, a pre-trained neural network, fine-tuned for oral disease classification:  

- 🌐 **Base Layers**: Pre-trained MobileNetV2 for feature extraction.  
- 📏 **Global Average Pooling**: Reduces feature maps for dense layers.  
- 🔠 **Fully Connected Layers**: Dense layers for classification.  
- 🛡️ **Dropout Regularization**: Prevents overfitting by random neuron deactivation.  

**Hyperparameters**:  
- **Epochs**: 20  
- **Batch Size**: 32  
- **Optimizer**: Adam  
- **Loss Function**: Categorical Crossentropy  

---

## 📊 Results  

- **Training Accuracy**: 🌟 ~97%  
- **Validation Accuracy**: 🌟 ~94%  

**Performance Graphs**:  
- Accuracy:  
![Training vs Validation Accuracy](link_to_accuracy_plot)  

- Loss:  
![Training vs Validation Loss](link_to_loss_plot)  

**Confusion Matrix**:  
Visual representation of model predictions vs true labels:  
![Confusion Matrix](link_to_confusion_matrix)  

**Classification Report**:  
| Class         | Precision | Recall | F1-Score |  
|---------------|-----------|--------|----------|  
| Healthy Gums  | 0.98      | 0.96   | 0.97     |  
| Cavities      | 0.94      | 0.93   | 0.94     |  
| Gum Disease   | 0.92      | 0.91   | 0.91     |  

---

## 🛠️ Installation  

1. **Clone the repository**:  
   ```bash
   git clone https://github.com/nikhil07897/Oral-Disease-Classification.git
   ```  

2. **Install required dependencies**:  
   ```bash
   pip install -r requirements.txt
   ```  

3. **Prepare the dataset**:  
   - Ensure images are stored in `TRAIN` and `TEST` directories.  

4. **Run the script**:  
   ```bash
   python oral_disease_classification.py
   ```  

---

## 🚀 How to Use  

- Add your custom dataset following the directory structure.  
- Modify hyperparameters or architecture as needed in `oral_disease_classification.py`.  
- Run the script to train and evaluate the model.  

**Example Command**:  
```bash
python oral_disease_classification.py
```  

---

## 🚀 Future Enhancements  

- 🔬 Add more disease categories for better coverage.  
- 📡 Deploy the model via a web app using Flask/Django.  
- 🧠 Integrate Grad-CAM for explainable AI in image predictions.  
- 📱 Create a mobile app for real-time oral disease detection.  

---

## 📢 Acknowledgments  

This project is powered by:  
- TensorFlow  
- Keras  
- OpenCV  
- MobileNetV2  
- Matplotlib  

Special thanks to the open-source community for providing datasets and tools! 🙌  

---

## 🤝 Contributors  

| 👨‍💻 Nikhil |  
| :---: |  
| [![GitHub followers](https://img.shields.io/github/followers/nikhil07897?style=social)](https://github.com/nikhil07897) |  
😊
