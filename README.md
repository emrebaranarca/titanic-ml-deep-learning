# Titanic Survival Prediction — ML & Deep Learning

A machine learning and deep learning project that predicts passenger survival on the Titanic using multiple models including a custom PyTorch neural network, Logistic Regression, Random Forest, and XGBoost. Built on the [Kaggle Titanic dataset](https://www.kaggle.com/competitions/titanic).

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Models](#models)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [License](#license)

## Overview

This project tackles the classic Kaggle Titanic challenge: *"predict which passengers survived the Titanic shipwreck."* It walks through the full data-science pipeline — from exploratory data analysis and feature engineering to training and comparing four different classification models, finishing with hyperparameter optimisation and a Kaggle-ready submission file.

## Dataset

The data comes from the [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic) and includes the following key features:

| Feature | Description |
|---------|-------------|
| `Pclass` | Ticket class (1st, 2nd, 3rd) |
| `Sex` | Passenger gender |
| `Age` | Age in years |
| `SibSp` | Number of siblings/spouses aboard |
| `Parch` | Number of parents/children aboard |
| `Fare` | Passenger fare |
| `Embarked` | Port of embarkation (C, Q, S) |
| `Cabin` | Cabin number |
| `Survived` | Target variable (0 = No, 1 = Yes) |

## Project Workflow

### 1. Exploratory Data Analysis (EDA)
- Dataset shape, types, and summary statistics
- Missing-value analysis with visualisations
- Survival distributions by gender, class, age, and embarkation port
- Correlation heatmap of numeric features

### 2. Feature Engineering
| New Feature | Logic |
|-------------|-------|
| `Title` | Extracted from passenger name (Mr, Miss, Mrs, Master, Rare) |
| `Family_size` | `SibSp + Parch + 1` |
| `Is_alone` | 1 if `Family_size == 1` |
| `Has_cabin` | 1 if cabin info exists |
| `Age_group` | Binned into Child, Young, Adult, MiddleAge, Senior |
| `Fare_category` | Quartile-based fare bins |

### 3. Data Preprocessing
- Missing age values filled using title-group medians
- Missing fare filled with median; missing embarked filled with mode
- Categorical variables one-hot encoded (`pd.get_dummies`)
- Features scaled with `StandardScaler`
- Stratified 80/20 train-test split

### 4. Model Training & Evaluation

Four models are trained and compared on Accuracy, F1-Score, AUC-ROC, and 5-fold Cross-Validation:

| Model | Description |
|-------|-------------|
| **PyTorch Neural Network** | Custom 2-hidden-layer network (64 → 32 neurons) with ReLU, Dropout, and Sigmoid output, trained for 200 epochs with Adam optimiser and BCE loss |
| **Logistic Regression** | Baseline linear model via scikit-learn |
| **Random Forest** | Ensemble of 200 trees (max depth 8) |
| **XGBoost** | Gradient-boosted trees (200 estimators, max depth 5) |

### 5. Hyperparameter Optimisation
- `GridSearchCV` over XGBoost parameters (`n_estimators`, `max_depth`, `learning_rate`, `min_child_weight`)
- 5-fold cross-validation to select the best configuration

### 6. Submission
- Best model applied to the test set with identical feature engineering and encoding
- Final predictions saved to `submission.csv`

## Models

### PyTorch Custom Neural Network
```
Input → Linear(64) → ReLU → Dropout(0.2) → Linear(32) → ReLU → Linear(1) → Sigmoid
```
- Loss: Binary Cross-Entropy
- Optimiser: Adam (lr = 0.01)
- Epochs: 200

### Scikit-learn / XGBoost Models
- **Logistic Regression** — `max_iter=1000`
- **Random Forest** — `n_estimators=200, max_depth=8, min_samples_split=5`
- **XGBoost** — `n_estimators=200, max_depth=5, learning_rate=0.1` (further tuned via GridSearchCV)

## Tech Stack

- **Python 3**
- **Data**: NumPy, Pandas
- **Visualisation**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn, XGBoost
- **Deep Learning**: PyTorch

## Getting Started

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost torch
```

### Running the Notebook

1. Download the Titanic dataset from [Kaggle](https://www.kaggle.com/competitions/titanic/data) and place `train.csv` and `test.csv` in the data directory.
2. Open and run the notebook:
   ```bash
   jupyter notebook who-live-titanic.ipynb
   ```
3. The notebook will train all models, display evaluation plots, and generate `submission.csv`.

## Project Structure

```
├── who-live-titanic.ipynb   # Main notebook (EDA → models → submission)
├── LICENSE                  # Apache 2.0
└── README.md                # This file
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).