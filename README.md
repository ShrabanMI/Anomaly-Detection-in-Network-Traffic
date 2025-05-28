# **Anomaly Detection in Network Traffic**

## **Project Overview**
This project focuses on detecting anomalies in network traffic using **machine learning (ML)** models. It compares the performance of two major ML approaches — **Ensemble Learning** and **Deep Learning** — to determine which is more effective for intrusion detection in cybersecurity.

The goal is to help researchers, developers, and cybersecurity professionals understand how different machine learning algorithms perform in identifying malicious activities in real-world network traffic data.

---

## **Key Features**
- **Anomaly Detection**: Detects unusual patterns in network traffic that may indicate cyberattacks.
- **Machine Learning Models Tested**:
  - **Ensemble Learning Models**: Random Forest, XGBoost, CatBoost, HistGradientBoosting
  - **Deep Learning Models**: LSTM, Multi-Layer Perceptron
- **Dataset Used**: BCCC-CIC-IDS2017
- **Performance Metrics**: Accuracy, Precision, Recall, and F1-Score

---

## **Dataset Summary**
- **Total Samples**: around 2.5 million samples
- **Features**: 122 (reduced to 117 after preprocessing)
- **Attack Types**: 12 attack classes + 1 benign class
- **Preprocessing Steps**:
  - Removal of irrelevant features
  - Label encoding of categorical features
  - Standardization of numerical features
  - Stratified train-test split (80-20)

---

## **Models Evaluated**
| Model | Type | Description |
|-------|------|-------------|
| **Random Forest** | Ensemble | Builds multiple decision trees and aggregates predictions |
| **XGBoost** | Ensemble | Gradient boosting algorithm with regularization |
| **CatBoost** | Ensemble | Efficiently handles categorical features |
| **HistGradientBoosting** | Ensemble | Fast training using histogram binning |
| **LSTM** | Deep Learning | Recurrent neural network for sequential data |
| **MLP** | Deep Learning | Feedforward neural network with hidden layers |

---

## **Results**
### **Best Performing Models**
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Random Forest** | 99.95% | 97.74% | 97.39% | 97.56% |
| **XGBoost** | 99.95% | 97.12% | 97.22% | 97.17% |

### **Deep Learning Models Performance**
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **LSTM** | 99.27% | 96.35% | 73.34% | 76.20% |
| **MLP** | 99.25% | 96.31% | 71.96% | 75.72% |

> **Observation**: Ensemble models outperformed deep learning models in terms of **recall and F1-score**, especially for minority attack classes. This indicates better generalization and reliability in detecting rare but critical threats.

---

## **Conclusion**
- **Ensemble Learning models (Random Forest, XGBoost)** are more effective for anomaly detection in structured network traffic data.
- They offer:
  - High accuracy
  - Better handling of class imbalance
  - Faster training
- **Deep Learning models (LSTM, MLP)**, while powerful, require more balanced and sequential data to perform well. Alongside that, model like LSTM will perform better in detecting sequential attacks.

---

## **Contributors**
- Md. Monowarul Islam  
- Fardous Nayeem  

---

## **How to Run on your machine**
To reproduce the results:
1. Clone this repository.
2. Download dataset. [https://drive.google.com/drive/folders/1JOLRaqb7fSOKuoutwTlEFtrai_y2M_tl?usp=drive_link]
3. Install required dependencies (pip install requirements.txt).
4. Run the Jupyter Notebooks or Python scripts provided.
5. Evaluate model performance using the given dataset.

---

## **References**
- Dataset: BCCC-CIC-IDS2017 [Behaviour-Centric Cybersecurity Center, Canadian Institute for Cyybersecurity, York University]
- Libraries used: Scikit-learn, XGBoost, CatBoost, TensorFlow, Pandas, NumPy  
- Tools: Jupyter Notebook, Python

--- 
