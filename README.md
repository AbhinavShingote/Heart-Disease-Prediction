# ❤️ Multimodal Heart Disease Prediction

## Group Horizon | MIT Academy of Engineering, Alandi, Pune
**Subject:** Predictive Analytics | Semester VI | 2025–26

### Team Members
| PRN | Roll No. | Name |
|-----|---------|------|
| 202301040145 | EDA41 | Pratik Shamrao Patil |
| 202301040140 | EDA39 | Abhinav Kishor Shingote |
| 202301040142 | EDA40 | Devarshi Nandeshwar |
| 202301040108 | EDA29 | Ravindra Nagtilak |
| 202301040109 | EDA30 | Prachit Kamble |

**Guided by:** Dr. Vaishali Wangikar | Ms. Shubhangi Kale

---

## 📌 Project Overview
A multimodal AI system for heart disease prediction using:
- Machine Learning (LR, RF, SVM, KNN, Decision Tree)
- Clustering (K-Means, DBSCAN, Hierarchical, K-Medoids)
- Deep Learning (ANN, CNN1D, LSTM, RNN/GRU, ResNet50)
- NLP (TF-IDF symptom analysis)
- Agentic AI (LangChain ReAct)
- Generative AI (Google Gemini API)

---

## 📊 Model Results

### Classification
| Model | Accuracy | F1-Score | AUC |
|-------|---------|----------|-----|
| KNN | 91.85% | 0.929 | 0.925 |
| Random Forest | 89.13% | 0.903 | 0.932 |
| SVM | 88.59% | 0.900 | 0.942 |

### Deep Learning
| Model | Accuracy |
|-------|---------|
| CNN1D | 98.62% |
| ResNet50 | 87.50% |
| RNN/GRU | 87.39% |
| LSTM | 81.11% |

---

## 📁 Project Structure

```
Heart_PA/
├── notebooks/
│   ├── 01_Classification_Clustering.ipynb
│   ├── 02_ANN_Tabular.ipynb
│   ├── 03_CNN1D_ECG.ipynb
│   ├── 04_LSTM_ECG.ipynb
│   ├── 05_RNN_GRU_ECG.ipynb
│   ├── 06_ResNet50_Xray.ipynb
│   └── 07_NLP_Symptoms.ipynb
├── models/          ← Model files (see Drive link below)
├── app.py           ← Streamlit dashboard
├── agentic_ai.py    ← LangChain Agent
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔗 Model Files (Google Drive)
Large `.h5` and `.pkl` files: [https://drive.google.com/drive/folders/1HAWWzmVB8p50UWtmd1Zmok07mh6-uJTV?usp=sharing]

---

## 🚀 How to Run

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run Streamlit app
```bash
streamlit run app.py
```

---

## 🛠️ Tech Stack
- Python 3.13
- TensorFlow / Keras
- Scikit-learn
- LangChain
- Google Gemini API
- Streamlit
- Pandas, NumPy, Matplotlib, Seaborn

---

## 📋 Datasets
- [Heart Disease Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- [MIT-BIH Heartbeat](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)
- [Chest X-ray](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)