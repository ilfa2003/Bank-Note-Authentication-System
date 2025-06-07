# 💸 Banknote Authentication using Machine Learning


## 📋 Project Overview

This project implements a machine learning-based system with comparative analysis of Five classifiers to authenticate banknotes and detect counterfeit currency using image processing and classification algorithms. The system analyzes wavelet-transformed features extracted from banknote images to determine authenticity with high accuracy.

## 📁 Project Structure

```
banknote-authentication/
├── implementation/
│   ├── BankNoteAuth.ipynb          # Main Jupyter notebook with ML implementation
│   ├── BankNote_Authentication.csv  # Dataset with extracted features
│   ├── app.py                      # Streamlit web application
│   ├── compare.png                 # Model comparison visualization
│   ├── image.jpeg                  # Sample banknote image
│   └── trained_model.pkl           # Serialized best performing model
├── BANKNOTE AUTHENTICATION.pptx    # Project presentation
├── ml report.docx                  # Detailed project report
└── bank note authentication.rar   # Compressed project files
```

## 📊 Dataset Information

- **Source**: Industrial camera images of banknote specimens. ([Data Link](https://www.kaggle.com/datasets/ritesaluja/bank-note-authentication-uci-data))
- **Resolution**: 400x400 pixels, grayscale, ~660 DPI
- **Feature Extraction**: Wavelet Transform applied to extract key features
- **Features**: Variance, Skewness, Kurtosis, and Entropy of wavelet coefficients
- **Target**: Binary classification (Authentic/Counterfeit)

## 🛠️ Implementation Pipeline

### 1. Data Preprocessing
- **Exploratory Data Analysis**: Using Seaborn and Matplotlib
- **Distribution Analysis**: Identified peaks in kurtosis and entropy features
- **Feature Scaling**: Normalization to prevent overfitting
- **Data Visualization**: Distribution plots and correlation analysis

### 2. Model Training & Selection
- **Cross-Validation**: K-fold validation for robust evaluation
- **Hyperparameter Tuning**: Grid search for optimal parameters
- **Performance Comparison**: Accuracy, precision, recall, and F1-score metrics
- **Model Selection**: SVM chosen as best performer

### 3. Model Evaluation
- **Test Set Performance**: Final evaluation on unseen data
- **Confusion Matrix**: Detailed classification results
- **ROC Curve Analysis**: Model discrimination capability
- **Feature Importance**: Understanding key predictive features

### 4. Web Application Development
- **Streamlit Interface**: User-friendly web application
- **Real-time Prediction**: Interactive sliders for feature input
- **Model Integration**: Pickle-serialized model deployment
- **Visualization**: Prediction confidence and results display

## 🚀 Getting Started

### Prerequisites
```bash
#Create an environemnt & intalll requirements.txt

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

### Running the Application
```bash
# Navigate to implementation directory
cd implementation/

# Run Streamlit app
streamlit run app.py
```

### Using the Jupyter Notebook
```bash
# Launch Jupyter Notebook
jupyter notebook BankNoteAuth.ipynb
```
## 📊 Model Performance

| Model              | Accuracy (%) |
|-------------------|--------------|
| Logistic Regression | 98.96     |
| Decision Tree | 97.92        |
| Random Forests | 98.96        |
| AdaBoost | 98.33        |
| SVM | **99.06**        |

*SVM achieved **97.8%** on the test dataset.*

---

**⭐ If you found this project helpful, please consider giving it a star!**
