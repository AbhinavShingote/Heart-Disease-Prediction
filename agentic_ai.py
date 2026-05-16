# ============================================================
# LANGCHAIN AGENTIC AI — HEART DISEASE PREDICTION
# Group: Horizon | MIT Academy of Engineering
# Models: LR, RF, SVM, KNN, DT, ANN, CNN1D, LSTM, RNN, ResNet50
# ============================================================

import os
import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# ============================================================
# PATHS
# ============================================================
MODELS_PATH = r"C:\Users\ABHINAV\Desktop\Heart_PA\models"

# ============================================================
# LOAD ALL MODELS
# ============================================================
def load_all_models():
    models = {}

    # ── Deep Learning ──
    try:
        models['ann']    = load_model(
            os.path.join(MODELS_PATH, 'ann_heart_disease_model.h5'))
        print("✅ ANN loaded")
    except Exception as e:
        print(f"❌ ANN: {e}")

    try:
        models['cnn1d']  = load_model(
            os.path.join(MODELS_PATH, 'cnn1d_ecg_model.h5'))
        print("✅ CNN1D loaded")
    except Exception as e:
        print(f"❌ CNN1D: {e}")

    try:
        models['lstm']   = load_model(
            os.path.join(MODELS_PATH, 'lstm_ecg_model.h5'))
        print("✅ LSTM loaded")
    except Exception as e:
        print(f"❌ LSTM: {e}")

    try:
        models['rnn']    = load_model(
            os.path.join(MODELS_PATH, 'rnn_ecg_model.h5'))
        print("✅ RNN loaded")
    except Exception as e:
        print(f"❌ RNN: {e}")

    try:
        models['resnet'] = load_model(
            os.path.join(MODELS_PATH, 'resnet50_xray_model.h5'))
        print("✅ ResNet50 loaded")
    except Exception as e:
        print(f"❌ ResNet50: {e}")

    # ── ML Classification ──
    ml_files = {
        'lr'  : 'logistic_model.pkl',
        'rf'  : 'rf_model.pkl',
        'svm' : 'svm_model.pkl',
        'knn' : 'knn_model.pkl',
        'dt'  : 'dt_model.pkl',
    }
    for key, fname in ml_files.items():
        try:
            with open(os.path.join(MODELS_PATH, fname), 'rb') as f:
                models[key] = pickle.load(f)
            print(f"✅ {key.upper()} loaded")
        except Exception as e:
            print(f"❌ {key.upper()}: {e}")

    # ── Supporting Files ──
    support_files = {
        'scaler'         : 'scaler.pkl',
        'kmeans'         : 'kmeans_model.pkl',
        'nlp'            : 'nlp_model.pkl',
        'vectorizer'     : 'tfidf_vectorizer.pkl',
        'feature_columns': 'feature_columns.pkl',
    }
    for key, fname in support_files.items():
        try:
            with open(os.path.join(MODELS_PATH, fname), 'rb') as f:
                models[key] = pickle.load(f)
            print(f"✅ {key} loaded")
        except Exception as e:
            print(f"❌ {key}: {e}")

    return models

# Global models
print("🚀 Loading all models...")
MODELS = load_all_models()
print(f"\n✅ Total models loaded: {len(MODELS)}")

# ============================================================
# TRAINED ACCURACIES (from your notebooks)
# ============================================================
TRAINED_ACC = {
    'Logistic Regression': 88.59,
    'Random Forest'      : 89.13,
    'SVM'                : 88.59,
    'KNN'                : 91.85,
    'Decision Tree'      : 79.89,
    'ANN'                : 87.50,
    'CNN1D'              : 98.62,
    'LSTM'               : 81.11,
    'RNN/GRU'            : 87.39,
    'ResNet50'           : 87.50,
}

# ============================================================
# HELPER — Build 20-feature input
# ============================================================
def build_input_array(data_dict):
    input_data = {
        'Age'               : data_dict['Age'],
        'RestingBP'         : data_dict['RestingBP'],
        'Cholesterol'       : data_dict['Cholesterol'],
        'FastingBS'         : data_dict['FastingBS'],
        'MaxHR'             : data_dict['MaxHR'],
        'Oldpeak'           : data_dict['Oldpeak'],
        'Sex_F'             : 1 if data_dict['Sex'] == 'Female' else 0,
        'Sex_M'             : 1 if data_dict['Sex'] == 'Male'   else 0,
        'ChestPainType_ASY' : 1 if data_dict['ChestPainType'] == 'ASY' else 0,
        'ChestPainType_ATA' : 1 if data_dict['ChestPainType'] == 'ATA' else 0,
        'ChestPainType_NAP' : 1 if data_dict['ChestPainType'] == 'NAP' else 0,
        'ChestPainType_TA'  : 1 if data_dict['ChestPainType'] == 'TA'  else 0,
        'RestingECG_LVH'    : 1 if data_dict['RestingECG'] == 'LVH'    else 0,
        'RestingECG_Normal' : 1 if data_dict['RestingECG'] == 'Normal' else 0,
        'RestingECG_ST'     : 1 if data_dict['RestingECG'] == 'ST'     else 0,
        'ExerciseAngina_N'  : 1 if data_dict['ExerciseAngina'] == 'No'  else 0,
        'ExerciseAngina_Y'  : 1 if data_dict['ExerciseAngina'] == 'Yes' else 0,
        'ST_Slope_Down'     : 1 if data_dict['ST_Slope'] == 'Down' else 0,
        'ST_Slope_Flat'     : 1 if data_dict['ST_Slope'] == 'Flat' else 0,
        'ST_Slope_Up'       : 1 if data_dict['ST_Slope'] == 'Up'   else 0,
    }
    feature_order = MODELS['feature_columns']
    input_array   = np.array([[input_data[col]
                                for col in feature_order]])
    return input_array


def get_model_explanation(model_name):
    explanations = {
        'SVM'               : 'SVM achieved the best ROC-AUC (0.9420) with 88.59% accuracy by finding an optimal separating boundary.',
        'Random Forest'     : 'Random Forest (89.13%) uses 100 decision trees, reducing overfitting and improving robustness.',
        'ANN'               : 'ANN (87.50%) captures non-linear patterns through multiple hidden layers.',
        'Logistic Regression': 'Logistic Regression (88.59%) provides interpretable probability scores and a strong baseline.',
        'KNN'               : 'KNN (91.85%) finds similar patients in training data and achieved the best classification accuracy.',
        'Decision Tree'     : 'Decision Tree (79.89%) provides clear rules but had lower accuracy than ensemble/distance models.',
        'CNN1D'             : 'CNN1D (98.62%) detects local ECG patterns like QRS complex. Best for signals.',
        'LSTM'              : 'LSTM (81.11%) captures long-term dependencies in ECG sequences.',
        'RNN/GRU'           : 'GRU (87.39%) is faster than LSTM with similar performance on ECG data.',
        'ResNet50'          : 'ResNet50 (87.50%) uses Transfer Learning from ImageNet. Best for X-ray images.',
    }
    return explanations.get(model_name, 'Strong performance on this dataset.')


# ============================================================
# TOOL 1 — TABULAR PREDICTION (LR + RF + SVM + KNN + DT + ANN)
# ============================================================
@tool
def predict_tabular_tool(input_str: str) -> str:
    """
    Use this tool when user provides clinical patient data like
    Age, Blood Pressure, Cholesterol, Heart Rate, Sex, etc.
    Input format (comma separated):
    age,resting_bp,cholesterol,fasting_bs,max_hr,oldpeak,sex,chest_pain,resting_ecg,exercise_angina,st_slope
    Example: 55,140,250,0,150,1.5,Male,ATA,Normal,No,Up
    This tool runs ALL 6 models: LR, RF, SVM, KNN, DT, ANN
    and recommends the best one using majority voting.
    """
    try:
        # Extract comma-separated values from input, handling extra text
        import re
        # Find pattern like: numbers,strings separated by commas
        comma_pattern = r'[\d.]+(?:,[\d.]+)*[,\w]+(?:,[A-Za-z]+)*'
        matches = re.findall(comma_pattern, input_str)
        
        if matches:
            # Take the first match that looks like our data
            candidate = matches[0]
            parts = [p.strip() for p in candidate.split(',')]
        else:
            # Fallback to simple split
            parts = [p.strip() for p in input_str.split(',')]
        
        # Filter out any non-numeric/non-categorical parts at the beginning
        while parts and not (parts[0].replace('.', '').isdigit() or parts[0] in ['Male', 'Female']):
            parts.pop(0)
        
        if len(parts) != 11:
            return f"Feature mismatch: Expected 11 features, got {len(parts)}. Input: '{input_str}'"

        # Validate categorical values
        valid_sex = ['Male', 'Female']
        valid_cp = ['ASY', 'ATA', 'NAP', 'TA']
        valid_ecg = ['Normal', 'ST', 'LVH']
        valid_angina = ['Yes', 'No']
        valid_slope = ['Up', 'Flat', 'Down']
        
        if parts[6] not in valid_sex:
            return f"Invalid Sex value '{parts[6]}'. Must be one of: {', '.join(valid_sex)}"
        if parts[7] not in valid_cp:
            return f"Invalid ChestPainType value '{parts[7]}'. Must be one of: {', '.join(valid_cp)}"
        if parts[8] not in valid_ecg:
            return f"Invalid RestingECG value '{parts[8]}'. Must be one of: {', '.join(valid_ecg)}"
        if parts[9] not in valid_angina:
            return f"Invalid ExerciseAngina value '{parts[9]}'. Must be one of: {', '.join(valid_angina)}"
        if parts[10] not in valid_slope:
            return f"Invalid ST_Slope value '{parts[10]}'. Must be one of: {', '.join(valid_slope)}"

        data_dict = {
            'Age'           : float(parts[0]),
            'RestingBP'     : float(parts[1]),
            'Cholesterol'   : float(parts[2]),
            'FastingBS'     : float(parts[3]),
            'MaxHR'         : float(parts[4]),
            'Oldpeak'       : float(parts[5]),
            'Sex'           : parts[6],
            'ChestPainType' : parts[7],
            'RestingECG'    : parts[8],
            'ExerciseAngina': parts[9],
            'ST_Slope'      : parts[10],
        }

        input_array  = build_input_array(data_dict)
        scaled_input = MODELS['scaler'].transform(input_array)

        results = {}

        # ── 1. Logistic Regression ──
        if 'lr' in MODELS:
            pred = MODELS['lr'].predict(input_array)[0]
            try:
                prob = MODELS['lr'].predict_proba(input_array)[0]
                conf = float(max(prob))
            except:
                conf = TRAINED_ACC['Logistic Regression'] / 100
            results['Logistic Regression'] = {
                'prediction'      : int(pred),
                'confidence'      : conf,
                'result'          : 'Heart Disease' if pred == 1 else 'No Disease',
                'trained_accuracy': TRAINED_ACC['Logistic Regression']
            }

        # ── 2. Random Forest ──
        if 'rf' in MODELS:
            pred = MODELS['rf'].predict(input_array)[0]
            try:
                prob = MODELS['rf'].predict_proba(input_array)[0]
                conf = float(max(prob))
            except:
                conf = TRAINED_ACC['Random Forest'] / 100
            results['Random Forest'] = {
                'prediction'      : int(pred),
                'confidence'      : conf,
                'result'          : 'Heart Disease' if pred == 1 else 'No Disease',
                'trained_accuracy': TRAINED_ACC['Random Forest']
            }

        # ── 3. SVM ──
        if 'svm' in MODELS:
            pred = MODELS['svm'].predict(input_array)[0]
            try:
                prob = MODELS['svm'].predict_proba(input_array)[0]
                conf = float(max(prob))
            except:
                conf = TRAINED_ACC['SVM'] / 100
            results['SVM'] = {
                'prediction'      : int(pred),
                'confidence'      : conf,
                'result'          : 'Heart Disease' if pred == 1 else 'No Disease',
                'trained_accuracy': TRAINED_ACC['SVM']
            }

        # ── 4. KNN ──
        if 'knn' in MODELS:
            pred = MODELS['knn'].predict(input_array)[0]
            try:
                prob = MODELS['knn'].predict_proba(input_array)[0]
                conf = float(max(prob))
            except:
                conf = TRAINED_ACC['KNN'] / 100
            results['KNN'] = {
                'prediction'      : int(pred),
                'confidence'      : conf,
                'result'          : 'Heart Disease' if pred == 1 else 'No Disease',
                'trained_accuracy': TRAINED_ACC['KNN']
            }

        # ── 5. Decision Tree ──
        if 'dt' in MODELS:
            pred = MODELS['dt'].predict(input_array)[0]
            try:
                prob = MODELS['dt'].predict_proba(input_array)[0]
                conf = float(max(prob))
            except:
                conf = TRAINED_ACC['Decision Tree'] / 100
            results['Decision Tree'] = {
                'prediction'      : int(pred),
                'confidence'      : conf,
                'result'          : 'Heart Disease' if pred == 1 else 'No Disease',
                'trained_accuracy': TRAINED_ACC['Decision Tree']
            }

        # ── 6. ANN ──
        if 'ann' in MODELS:
            prob = MODELS['ann'].predict(scaled_input, verbose=0)[0][0]
            pred = 1 if prob > 0.5 else 0
            conf = prob if pred == 1 else 1 - prob
            results['ANN'] = {
                'prediction'      : pred,
                'confidence'      : float(conf),
                'result'          : 'Heart Disease' if pred == 1 else 'No Disease',
                'trained_accuracy': TRAINED_ACC['ANN']
            }

        # ── KMeans Risk Level ──
        # KMeans was trained on original 6 numerical features
        numerical_features = input_array[:, :6]  # Age, RestingBP, Cholesterol, FastingBS, MaxHR, Oldpeak
        cluster    = MODELS['kmeans'].predict(numerical_features)[0]
        risk_map   = {0: 'Low Risk', 1: 'Medium Risk', 2: 'High Risk'}
        risk_level = risk_map.get(int(cluster), 'Medium Risk')

        # ── Majority Voting ──
        votes          = [r['prediction'] for r in results.values()]
        disease_votes  = votes.count(1)
        no_disease_votes = votes.count(0)
        majority       = 1 if disease_votes > no_disease_votes else 0
        majority_result = 'Heart Disease' if majority == 1 else 'No Disease'

        # ── Best Models ──
        best_by_acc  = max(results.items(),
                           key=lambda x: x[1]['trained_accuracy'])
        best_by_conf = max(results.items(),
                           key=lambda x: x[1]['confidence'])

        # ── Format Output ──
        output = f"""
TABULAR DATA — ALL 6 MODELS COMPARISON
========================================
Patient Info:
  Age={data_dict['Age']}, BP={data_dict['RestingBP']},
  Cholesterol={data_dict['Cholesterol']}, MaxHR={data_dict['MaxHR']},
  Sex={data_dict['Sex']}, ChestPain={data_dict['ChestPainType']},
  ST_Slope={data_dict['ST_Slope']}

Individual Model Results:
--------------------------
"""
        for name, res in results.items():
            marker = " ⭐" if name == best_by_acc[0] else ""
            output += (f"  {name:<22}: {res['result']:<14} | "
                       f"Confidence: {res['confidence']*100:5.1f}% | "
                       f"Trained: {res['trained_accuracy']:.1f}%{marker}\n")

        output += f"""
Voting Summary:
  Heart Disease votes : {disease_votes}/{len(votes)}
  No Disease votes    : {no_disease_votes}/{len(votes)}

Risk Level (K-Means Clustering): {risk_level}
Majority Vote Result           : {majority_result}

AGENT RECOMMENDATION:
━━━━━━━━━━━━━━━━━━━━━
✅ Best Model (Accuracy)   : {best_by_acc[0]} ({best_by_acc[1]['trained_accuracy']}%)
✅ Best Model (Confidence) : {best_by_conf[0]} ({best_by_conf[1]['confidence']*100:.1f}%)
✅ Final Decision           : {majority_result}
✅ Risk Category            : {risk_level}

Why {best_by_acc[0]}:
{get_model_explanation(best_by_acc[0])}

{'⚠️ ALERT: Heart disease detected! Consult a cardiologist immediately.' if majority == 1 else '✅ No strong disease indicators. Maintain healthy lifestyle.'}
"""
        return output

    except Exception as e:
        return f"Error in tabular prediction: {str(e)}. Format: age,bp,chol,fbs,maxhr,oldpeak,sex,cp,ecg,ea,slope"


# ============================================================
# TOOL 2 — ECG SIGNAL PREDICTION (CNN1D + LSTM + RNN)
# ============================================================
@tool
def predict_ecg_tool(input_str: str) -> str:
    """
    Use this tool when user provides ECG signal data.
    Input: path to CSV file with 187 ECG values, or 'sample' for demo.
    Example: C:/path/to/ecg_signal.csv
    Runs CNN1D, LSTM and RNN models and compares results.
    """
    try:
        if input_str.strip().lower() == 'sample':
            t          = np.linspace(0, 4*np.pi, 187)
            ecg_signal = (np.sin(t) + 0.3*np.sin(3*t) +
                          0.1*np.random.randn(187)).astype(float)
        elif os.path.exists(input_str.strip()):
            ecg_df     = pd.read_csv(input_str.strip(), header=None)
            ecg_signal = ecg_df.iloc[0, :187].values.astype(float)
        else:
            t          = np.linspace(0, 4*np.pi, 187)
            ecg_signal = np.sin(t).astype(float)

        if len(ecg_signal) >= 187:
            ecg_signal = ecg_signal[:187]
        else:
            ecg_signal = np.pad(ecg_signal,
                                (0, 187 - len(ecg_signal)),
                                mode='constant')

        signal_3d   = ecg_signal.reshape(1, 187, 1)
        class_names = {
            0: 'Normal', 1: 'Supraventricular',
            2: 'Ventricular', 3: 'Fusion', 4: 'Unknown'
        }
        risk_map = {
            'Normal'          : 'Low Risk',
            'Supraventricular': 'Medium Risk',
            'Ventricular'     : 'High Risk',
            'Fusion'          : 'High Risk',
            'Unknown'         : 'Medium Risk'
        }

        results = {}

        if 'cnn1d' in MODELS:
            prob  = MODELS['cnn1d'].predict(signal_3d, verbose=0)[0]
            cls   = class_names[np.argmax(prob)]
            results['CNN1D'] = {
                'class'     : cls,
                'confidence': float(np.max(prob)),
                'risk'      : risk_map[cls],
                'trained_acc': TRAINED_ACC['CNN1D']
            }

        if 'lstm' in MODELS:
            prob  = MODELS['lstm'].predict(signal_3d, verbose=0)[0]
            cls   = class_names[np.argmax(prob)]
            results['LSTM'] = {
                'class'     : cls,
                'confidence': float(np.max(prob)),
                'risk'      : risk_map[cls],
                'trained_acc': TRAINED_ACC['LSTM']
            }

        if 'rnn' in MODELS:
            prob  = MODELS['rnn'].predict(signal_3d, verbose=0)[0]
            cls   = class_names[np.argmax(prob)]
            results['RNN/GRU'] = {
                'class'     : cls,
                'confidence': float(np.max(prob)),
                'risk'      : risk_map[cls],
                'trained_acc': TRAINED_ACC['RNN/GRU']
            }

        # Majority vote
        classes     = [r['class'] for r in results.values()]
        final_class = max(set(classes), key=classes.count)
        final_risk  = risk_map[final_class]
        best_model  = max(results.items(),
                          key=lambda x: x[1]['trained_acc'])

        output = f"""
ECG SIGNAL ANALYSIS RESULTS
==============================
Signal Length: 187 timesteps

Individual Model Results:
--------------------------
"""
        for name, res in results.items():
            marker = " ⭐" if name == best_model[0] else ""
            output += (f"  {name:<10}: {res['class']:<20} | "
                       f"Conf: {res['confidence']*100:5.1f}% | "
                       f"Risk: {res['risk']:<12} | "
                       f"Trained: {res['trained_acc']}%{marker}\n")

        output += f"""
Majority Vote Class: {final_class}
Final Risk Level   : {final_risk}

AGENT RECOMMENDATION:
━━━━━━━━━━━━━━━━━━━━━
✅ Best Model    : {best_model[0]} ({best_model[1]['trained_acc']}% accuracy)
✅ ECG Class     : {final_class}
✅ Risk Level    : {final_risk}

{get_model_explanation(best_model[0])}

{'⚠️ Abnormal ECG detected. Consult a cardiologist immediately.' if final_class != 'Normal' else '✅ Normal ECG pattern detected. Heart rhythm appears healthy.'}
"""
        return output

    except Exception as e:
        return f"Error in ECG prediction: {str(e)}"


# ============================================================
# TOOL 3 — XRAY IMAGE PREDICTION (ResNet50)
# ============================================================
@tool
def predict_xray_tool(image_path: str) -> str:
    """
    Use this tool when user uploads or provides a chest X-ray image.
    Input: full path to image file (JPG or PNG).
    Example: C:/path/to/xray.jpg
    Uses ResNet50 Transfer Learning model.
    """
    try:
        image     = Image.open(image_path.strip())
        img       = image.resize((224, 224))
        img_array = img_to_array(img) / 255.0

        if img_array.shape[-1] == 1:
            img_array = np.concatenate([img_array]*3, axis=-1)
        elif img_array.shape[-1] == 4:
            img_array = img_array[:, :, :3]

        img_array = np.expand_dims(img_array, axis=0)

        pred_prob  = MODELS['resnet'].predict(img_array, verbose=0)[0][0]
        pred       = 1 if pred_prob > 0.5 else 0
        confidence = pred_prob if pred == 1 else 1 - pred_prob
        result     = 'PNEUMONIA Detected' if pred == 1 else 'NORMAL — No Disease'

        if pred_prob > 0.7:
            risk_level = 'High Risk'
        elif pred_prob > 0.4:
            risk_level = 'Medium Risk'
        else:
            risk_level = 'Low Risk'

        output = f"""
CHEST X-RAY ANALYSIS RESULTS
================================
Image Path : {image_path}
Model Used : ResNet50 (Transfer Learning — ImageNet pretrained)
Architecture: 50-layer Deep Residual Network

Result:
--------
  Prediction  : {result}
  Confidence  : {confidence*100:.1f}%
  Risk Level  : {risk_level}
  Trained Acc : {TRAINED_ACC['ResNet50']}%

AGENT RECOMMENDATION:
━━━━━━━━━━━━━━━━━━━━━
✅ Model Used  : ResNet50 (Best and only image model)
✅ Prediction  : {result}
✅ Risk        : {risk_level}

{get_model_explanation('ResNet50')}

{'⚠️ Abnormality detected in X-ray. Immediate medical consultation recommended.' if pred == 1 else '✅ X-ray appears normal. No obvious cardiac abnormality detected.'}
"""
        return output

    except Exception as e:
        return f"Error in X-ray prediction: {str(e)}"


# ============================================================
# TOOL 4 — NLP SYMPTOM PREDICTION
# ============================================================
@tool
def predict_symptoms_tool(symptoms: str) -> str:
    """
    Use this tool when user describes symptoms in plain English text.
    Input: symptom description in plain English.
    Example: chest pain shortness of breath fatigue sweating dizziness
    Returns risk level using NLP model.
    """
    try:
        text_vec   = MODELS['vectorizer'].transform([symptoms])
        risk_level = MODELS['nlp'].predict(text_vec)[0]
        pred_proba = MODELS['nlp'].predict_proba(text_vec)[0]
        classes    = MODELS['nlp'].classes_
        confidence = float(max(pred_proba))
        prediction = 1 if risk_level == 'High Risk' else 0

        # Class probabilities
        class_probs = dict(zip(classes, pred_proba))

        # Top keywords
        feature_names = MODELS['vectorizer'].get_feature_names_out()
        tfidf_scores  = text_vec.toarray()[0]
        top_idx       = tfidf_scores.argsort()[-5:][::-1]
        top_keywords  = [feature_names[i] for i in top_idx
                         if tfidf_scores[i] > 0]

        output = f"""
SYMPTOM ANALYSIS RESULTS
==========================
Input      : {symptoms[:100]}...
Model Used : NLP (TF-IDF Vectorizer + Logistic Regression)

Results:
---------
  Risk Level  : {risk_level}
  Confidence  : {confidence*100:.1f}%
  Prediction  : {'Heart Disease Risk' if prediction == 1 else 'Low/No Risk'}

Class Probabilities:
"""
        for cls, prob in class_probs.items():
            bar = '█' * int(prob * 20)
            output += f"  {cls:<15}: {prob*100:5.1f}% {bar}\n"

        output += f"""
Key Symptoms Detected: {', '.join(top_keywords) if top_keywords else 'General symptoms'}

AGENT RECOMMENDATION:
━━━━━━━━━━━━━━━━━━━━━
✅ Risk Level  : {risk_level}
✅ Confidence  : {confidence*100:.1f}%

{'🚨 HIGH RISK: Symptoms strongly indicate heart disease. Seek immediate medical attention.' if risk_level == 'High Risk' else '⚠️ MEDIUM RISK: Some concerning symptoms. Schedule a medical checkup soon.' if risk_level == 'Medium Risk' else '✅ LOW RISK: Symptoms do not strongly indicate heart disease. Maintain healthy lifestyle.'}
"""
        return output

    except Exception as e:
        return f"Error in symptom prediction: {str(e)}"


# ============================================================
# TOOL 5 — MODEL COMPARISON
# ============================================================
@tool
def compare_models_tool(data_type: str) -> str:
    """
    Use this tool to compare all available models for a data type.
    Input: one of these values - tabular, ecg, image, clustering, all
    Returns detailed comparison with recommendation.
    """
    data_type = data_type.lower().strip()

    if data_type == 'tabular':
        return """
TABULAR DATA - ALL 6 MODELS COMPARISON
======================================
Model                | Type          | Accuracy | Precision | Recall | F1     | Best?
---------------------|---------------|----------|-----------|--------|--------|-------
Decision Tree        | ML Rule-based | 79.89%   | 0.8218    | 0.8137 | 0.8177 | -
ANN                  | Deep Learning | 87.50%   | 0.88      | 0.88   | 0.87   | Best DL
Logistic Regression  | ML Linear     | 88.59%   | 0.8716    | 0.9314 | 0.9005 | -
SVM                  | ML Margin     | 88.59%   | 0.8716    | 0.9314 | 0.9005 | Best AUC
Random Forest        | ML Ensemble   | 89.13%   | 0.8942    | 0.9118 | 0.9029 | -
KNN                  | ML Distance   | 91.85%   | 0.8991    | 0.9608 | 0.9289 | BEST

RECOMMENDATION:
  Primary   : KNN (91.85%) - Highest classification accuracy
  Best AUC  : SVM (ROC-AUC 0.9420) - Strong ranking performance
  Secondary : Random Forest (89.13%) - Robust ensemble
  Deep Learn: ANN (87.50%) - Best non-linear tabular neural model
  Majority Vote of all 6 models gives most reliable prediction.
"""

    elif data_type == 'ecg':
        return """
ECG SIGNAL - 3 DEEP LEARNING MODELS COMPARISON
==============================================
Model    | Architecture      | Accuracy | Strength             | Best?
---------|-------------------|----------|----------------------|-------
LSTM     | LSTM layers       | 81.11%   | Long-term memory     | -
RNN/GRU  | GRU layers        | 87.39%   | Fast sequence        | -
CNN1D    | Conv1D + MaxPool  | 98.62%   | Local ECG patterns   | BEST

RECOMMENDATION:
  CNN1D is clearly best (98.62%) for ECG signal classification.
  It detects local QRS complex, P-wave, T-wave patterns efficiently.
  Use CNN1D as primary, RNN/GRU as secondary for cross-validation.
"""

    elif data_type == 'image':
        return """
IMAGE DATA - CHEST X-RAY MODEL
==============================
Model    | Approach          | Accuracy | Precision | Recall | F1   | Best?
---------|-------------------|----------|-----------|--------|------|-------
ResNet50 | Transfer Learning | 87.50%   | 0.88      | 0.88   | 0.88 | BEST

RECOMMENDATION:
  ResNet50 uses transfer learning from ImageNet and achieved 87.50% test accuracy.
  NORMAL F1 is 0.83 and PNEUMONIA F1 is 0.90 in the notebook output.
"""

    elif data_type == 'clustering':
        return """
CLUSTERING METHODS COMPARISON
=============================
Method        | Type            | Clusters | Silhouette | Davies-Bouldin | Calinski-Harabasz | Best?
--------------|-----------------|----------|------------|----------------|-------------------|------
Hierarchical  | Linkage-based   | 3        | 0.2851     | 1.2444         | 936.06            | -
K-Medoids     | Medoid-based    | 3        | 0.3251     | 1.1737         | 1006.38           | -
K-Means       | Centroid-based  | 3        | 0.3291     | 1.1684         | 1015.92           | Used for risk levels
DBSCAN        | Density-based   | 2        | 0.5930     | 0.6623         | 1387.21           | BEST METRICS

RECOMMENDATION:
  DBSCAN gives the best clustering metrics in the updated notebook.
  K-Means with K=3 is still used for patient risk stratification
  because the app needs exactly Low, Medium and High risk groups.
"""

    elif data_type == 'all':
        return """
MASTER COMPARISON - ALL MODELS ALL DATA TYPES
=============================================
Data Type  | Best Model          | Accuracy / Metric | Runner Up / Note
-----------|---------------------|-------------------|------------------
Tabular    | KNN                 | 91.85% accuracy   | SVM best ROC-AUC 0.9420
ECG        | CNN1D               | 98.62% accuracy   | RNN/GRU 87.39%
Image      | ResNet50            | 87.50% accuracy   | Transfer learning
Symptoms   | TF-IDF + Logistic Regression | 93.33% accuracy | Macro F1 0.93
Clustering | DBSCAN              | Best metrics      | K-Means used for 3 risk levels

OVERALL SYSTEM BEST MODELS:
  1. CNN1D  -> 98.62% (ECG Signal)
  2. NLP    -> 93.33% (Symptoms Text)
  3. KNN    -> 91.85% (Tabular Clinical)
  4. ResNet50 -> 87.50% (Chest X-ray)

SYSTEM APPROACH:
  Agentic AI routes each input to its specialist model.
  Multiple models run for tabular and ECG data.
  Majority voting and comparison improve reliability.
"""

    else:
        return "Please specify: tabular, ecg, image, clustering, or all"


# ============================================================
# CREATE LANGCHAIN REACT AGENT
# ============================================================
def create_heart_disease_agent(gemini_api_key: str):
    """Creates LangChain ReAct Agent for Heart Disease Prediction"""

    model_name = "gemini-2.5-flash"
    print(f"[DEBUG] Gemini model set to: {model_name}")
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        api_key=gemini_api_key,
        temperature=0
    )

    tools = [
        predict_tabular_tool,
        predict_ecg_tool,
        predict_xray_tool,
        predict_symptoms_tool,
        compare_models_tool,
    ]

    agent_prompt = PromptTemplate.from_template("""
You are an intelligent Medical AI Agent for Heart Disease Prediction
developed by Group Horizon at MIT Academy of Engineering.

Your capabilities:
1. Analyze clinical tabular data using 6 ML/DL models
2. Analyze ECG signals using CNN1D, LSTM, RNN models
3. Analyze chest X-ray images using ResNet50
4. Analyze symptoms using NLP model
5. Compare models and recommend the best one

Instructions:
- Automatically detect input type from user query
- Select the appropriate tool without asking
- Run prediction and compare all available models
- Recommend the best model with clear reasoning
- Provide majority-vote based final prediction
- Give health advice (always recommend consulting a doctor)
- Be concise, accurate and helpful

Available Tools: {tools}
Tool Names: {tool_names}

Format:
Question: input question
Thought: What type of input is this? Which tool should I use?
Action: tool_name
Action Input: properly formatted input for the tool
Observation: tool result
Thought: What does this result mean? Is more analysis needed?
Final Answer: Clear prediction + best model + risk level + health advice

Begin!

Question: {input}
{agent_scratchpad}
""")

    agent = create_react_agent(llm, tools, agent_prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        max_iterations=6,
        handle_parsing_errors=True,
        return_intermediate_steps=True
    )

    return agent_executor


# ============================================================
# DIRECT PREDICTION (Used by Streamlit — No Agent Overhead)
# ============================================================
def direct_predict_tabular(data_dict):
    """Direct prediction without LangChain for Streamlit speed"""
    try:
        input_array  = build_input_array(data_dict)
        scaled_input = MODELS['scaler'].transform(input_array)

        results = {}

        ml_models = [
            ('Logistic Regression', 'lr'),
            ('Random Forest', 'rf'),
            ('SVM', 'svm'),
            ('KNN', 'knn'),
            ('Decision Tree', 'dt'),
        ]

        for name, key in ml_models:
            if key in MODELS:
                pred = MODELS[key].predict(input_array)[0]
                try:
                    prob = MODELS[key].predict_proba(input_array)[0]
                    conf = float(max(prob))
                except:
                    conf = TRAINED_ACC.get(name, 80) / 100
                results[name] = {
                    'prediction'      : int(pred),
                    'confidence'      : conf,
                    'result'          : 'Heart Disease' if pred == 1 else 'No Disease',
                    'trained_accuracy': TRAINED_ACC.get(name, 80)
                }

        if 'ann' in MODELS:
            prob = MODELS['ann'].predict(scaled_input, verbose=0)[0][0]
            pred = 1 if prob > 0.5 else 0
            conf = prob if pred == 1 else 1 - prob
            results['ANN'] = {
                'prediction'      : pred,
                'confidence'      : float(conf),
                'result'          : 'Heart Disease' if pred == 1 else 'No Disease',
                'trained_accuracy': TRAINED_ACC['ANN']
            }

        cluster    = MODELS['kmeans'].predict(input_array[:, :6])[0]  # Use only numerical features
        risk_map   = {0: 'Low Risk', 1: 'Medium Risk', 2: 'High Risk'}
        risk_level = risk_map.get(int(cluster), 'Medium Risk')

        votes     = [r['prediction'] for r in results.values()]
        majority  = 1 if votes.count(1) > votes.count(0) else 0
        best_acc  = max(results.items(),
                        key=lambda x: x[1]['trained_accuracy'])
        confidence = best_acc[1]['confidence']

        return majority, confidence, risk_level, results

    except Exception as e:
        return None, None, None, {}


def direct_predict_ecg(ecg_signal):
    """Direct ECG prediction for Streamlit"""
    try:
        if len(ecg_signal) >= 187:
            ecg_signal = ecg_signal[:187]
        else:
            ecg_signal = np.pad(ecg_signal,
                                (0, 187 - len(ecg_signal)),
                                mode='constant')

        signal_3d = ecg_signal.reshape(1, 187, 1)
        class_names = {
            0: 'Normal', 1: 'Supraventricular',
            2: 'Ventricular', 3: 'Fusion', 4: 'Unknown'
        }
        risk_map = {
            'Normal': 'Low Risk',
            'Supraventricular': 'Medium Risk',
            'Ventricular': 'High Risk',
            'Fusion': 'High Risk',
            'Unknown': 'Medium Risk'
        }

        results = {}

        for name, key, acc_key in [
            ('CNN1D',   'cnn1d', 'CNN1D'),
            ('LSTM',    'lstm',  'LSTM'),
            ('RNN/GRU', 'rnn',   'RNN/GRU'),
        ]:
            if key in MODELS:
                prob = MODELS[key].predict(signal_3d, verbose=0)[0]
                cls  = class_names[np.argmax(prob)]
                results[name] = {
                    'class'      : cls,
                    'confidence' : float(np.max(prob)),
                    'risk'       : risk_map[cls],
                    'trained_acc': TRAINED_ACC[acc_key]
                }

        classes     = [r['class'] for r in results.values()]
        final_class = max(set(classes), key=classes.count)
        final_risk  = risk_map[final_class]
        prediction  = 0 if final_class == 'Normal' else 1
        confidence  = float(np.mean([r['confidence']
                                     for r in results.values()]))

        return prediction, confidence, final_risk, results

    except Exception as e:
        return None, None, None, {}


def direct_predict_image(image):
    """Direct image prediction for Streamlit"""
    try:
        img       = image.resize((224, 224))
        img_array = img_to_array(img) / 255.0

        if img_array.shape[-1] == 1:
            img_array = np.concatenate([img_array]*3, axis=-1)
        elif img_array.shape[-1] == 4:
            img_array = img_array[:, :, :3]

        img_array  = np.expand_dims(img_array, axis=0)
        pred_prob  = MODELS['resnet'].predict(img_array, verbose=0)[0][0]
        pred       = 1 if pred_prob > 0.5 else 0
        confidence = pred_prob if pred == 1 else 1 - pred_prob
        risk_level = ('High Risk'   if pred_prob > 0.7 else
                      'Medium Risk' if pred_prob > 0.4 else
                      'Low Risk')

        return pred, float(confidence), risk_level

    except Exception as e:
        return None, None, None


def direct_predict_symptoms(text):
    """Direct NLP prediction for Streamlit"""
    try:
        text_vec   = MODELS['vectorizer'].transform([text])
        risk_level = MODELS['nlp'].predict(text_vec)[0]
        pred_proba = MODELS['nlp'].predict_proba(text_vec)[0]
        confidence = float(max(pred_proba))
        prediction = 1 if risk_level == 'High Risk' else 0
        return prediction, confidence, risk_level
    except Exception as e:
        return None, None, None
