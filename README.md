# 🌾 Predicting Irrigation Need

A machine learning project that classifies agricultural irrigation demand into **Low**, **Medium**, or **High** categories using environmental and agronomic features. The final solution is a soft-voting ensemble of LightGBM, XGBoost, and CatBoost.

---

[Demo App](https://irrigation-need-classification-zyivtfewdtbbct3drupvpg.streamlit.app/)

## 📋 Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Feature Engineering](#feature-engineering)
- [Modeling](#modeling)
- [Results](#results)
- [Saved Artifacts](#saved-artifacts)
- [Tech Stack](#tech-stack)
- [How to Run](#how-to-run)

---

## Problem Statement

Efficient irrigation scheduling is critical for sustainable agriculture. Over-irrigation wastes water; under-irrigation damages crops. This project builds a multi-class classifier that predicts whether a field's irrigation need is **Low**, **Medium**, or **High** given soil, weather, crop, and farm management data — enabling smarter, data-driven irrigation decisions.

---

## Dataset

| Split | Rows | Notes |
|---|---|---|
| Train | — | Loaded from `train.csv` |

**Target variable:** `Irrigation_Need` (imbalanced — Low ≈ 59%, Medium ≈ 38%, High ≈ 3%)

### Features

**Numerical**

| Feature | Description |
|---|---|
| `Soil_Moisture` | Current moisture level in the soil |
| `Rainfall_mm` | Recent rainfall in millimetres |
| `Temperature_C` | Ambient temperature in °C |
| `Humidity` | Relative humidity (%) |
| `Sunlight_Hours` | Daily sunlight hours |
| `Wind_Speed_kmh` | Wind speed in km/h |
| `Previous_Irrigation_mm` | Water applied in the previous irrigation cycle |
| `Electrical_Conductivity` | Soil salinity indicator |
| `Organic_Carbon` | Soil organic carbon content |
| `Field_Area_hectare` | Size of the field |

**Categorical**

| Feature | Description |
|---|---|
| `Soil_Type` | Type of soil (e.g., Loamy, Sandy, Clay) |
| `Crop_Type` | Crop being grown (e.g., Sugarcane, Rice, Wheat) |
| `Crop_Growth_Stage` | Current growth stage (Seedling, Flowering, Harvesting, etc.) |
| `Season` | Agricultural season |
| `Irrigation_Type` | Method used (e.g., Drip, Sprinkler, Flood) |
| `Water_Source` | Source of irrigation water |
| `Region` | Geographic region |
| `Mulching_Used` | Whether mulching is applied (Yes/No) |

---

## Project Structure

```
irrigation-need-prediction/
│
├── Irrigation_need_analysis.ipynb   # Main notebook (EDA → FE → Modeling)
│
└── app/
    ├── irrigation_app.pkl           # All artifacts bundled
    ├── irrigation_ensemble_model.pkl
    ├── label_encoders.pkl
    ├── selected_features.pkl
    ├── target_mapping.pkl
    └── metrics.pkl
    └── app.py
```

---

## Exploratory Data Analysis

### Data Quality
- **No missing values** and **no duplicates** in the training set.
- **No significant outliers** detected via the IQR method across all numeric features.
- Features follow approximately uniform distributions, suggesting the dataset may be synthetic or heavily preprocessed.

### Key Findings

**Feature Influence on Irrigation Need (from KDE class-separation analysis)**

| Feature | Influence Strength |
|---|---|
| Soil_Moisture | ⭐⭐⭐⭐⭐ Very Strong |
| Temperature_C | ⭐⭐⭐⭐ Strong |
| Rainfall_mm | ⭐⭐⭐⭐ Strong |
| Wind_Speed_kmh | ⭐⭐⭐ Moderate |
| Humidity | ⭐⭐ Weak–Moderate |
| Previous_Irrigation_mm | ⭐ Weak |
| Sunlight_Hours | ⭐ Weak |

**Decision boundary (derived from scatter analysis):**

$$\text{Irrigation Need} = \begin{cases} \text{Low} & \text{if Soil Moisture} > 25 \\ \text{Medium} & \text{if Soil Moisture} \le 25 \text{ AND Temp} < 28°C \\ \text{High/Medium Mix} & \text{if Soil Moisture} \le 25 \text{ AND Temp} \ge 28°C \end{cases}$$

**Soil & Crop Insights**
- **Sandy soil** retains the least moisture; **Loamy soil** retains the most.
- **Sugarcane** is the most water-intensive crop (highest avg. irrigation volume).
- **Mulching** significantly reduces High irrigation demand (from ~5.85% → ~0.79%).
- The **Seedling** growth stage requires the most water per cycle.

**Irrigation Method Efficiency**
- **Drip irrigation** uses the least water volume while maintaining the highest soil moisture — making it the most efficient method.

---

## Feature Engineering

Three domain-informed features were constructed to capture water stress dynamics:

| Engineered Feature | Formula | Rationale |
|---|---|---|
| `ET_Proxy` | `(Temperature × Sunlight_Hours × Wind_Speed) / (Humidity + 1)` | Proxy for evapotranspiration — how fast the environment dries out the soil |
| `Water_Stress_Index` | `ET_Proxy / (Soil_Moisture + 1)` | Combines atmospheric demand with current soil water availability |
| `Rain_Temp_Ratio` | `Rainfall_mm / (Temperature_C + 1)` | Balances precipitation against temperature-driven water loss |

All three features showed statistically significant separation across irrigation classes (ANOVA p-value ≈ 0).

### Feature Selection

Three methods were combined — **ANOVA F-score**, **Pearson correlation with target**, and **Mutual Information** — to build a final ranked list. Three low-signal features were dropped: `Sunlight_Hours`, `Field_Area_hectare`, and `Organic_Carbon`.

**Final 12 selected features:**

```
Soil_Moisture, Water_Stress_Index, Rainfall_mm, Temperature_C,
Wind_Speed_kmh, Rain_Temp_Ratio, ET_Proxy, Humidity,
Previous_Irrigation_mm, Electrical_Conductivity,
Crop_Growth_Stage, Mulching_Used
```

Categorical features were encoded with `LabelEncoder` prior to modeling.

---

## Modeling

### Cross-Validation Baseline (5-Fold Stratified KFold)

| Configuration | Accuracy | Balanced Accuracy | Macro F1 |
|---|---|---|---|
| Model A — All Features | 0.984 | 0.961 | 0.969 |
| Model B — Selected Features | 0.984 | 0.961 | 0.969 |

### Individual Models Evaluated

| Model | Library |
|---|---|
| LightGBM | `lightgbm` |
| XGBoost | `xgboost` |
| CatBoost | `catboost` |
| Random Forest | `sklearn` |

All gradient boosting models used: `n_estimators=300`, `learning_rate=0.05`, `random_state=42`.

### Final Model — Soft Voting Ensemble

```python
VotingClassifier(
    estimators=[('lgbm', lgbm_model), ('xgb', model_xgb), ('cat_boost', model_cat)],
    voting='soft'
)
```

The ensemble averages class probability outputs from all three boosting models before making a final prediction.

---

## Results

Evaluated on a stratified 20% holdout test set:

| Metric | Score |
|---|---|
| Accuracy | 0.984 |
| Balanced Accuracy | 0.960 |
| Macro F1 | 0.969 |
| OVR ROC-AUC | 0.997 |


**ROC curves** were plotted per class (One-vs-Rest), with Class 2 (High) being the most challenging due to class imbalance (~3% of samples).

---

## Saved Artifacts

All model artifacts are saved with `joblib` for downstream inference or deployment:

| File | Contents |
|---|---|
| `irrigation_ensemble_model.pkl` | Trained VotingClassifier |
| `label_encoders.pkl` | Dict of `LabelEncoder` objects per categorical column |
| `selected_features.pkl` | Ordered list of 12 input features |
| `target_mapping.pkl` | `{0: "Low", 1: "Medium", 2: "High"}` |
| `metrics.pkl` | Dict of final evaluation scores |
| `irrigation_app.pkl` | All of the above bundled into one artifact |

**Loading the bundle:**
```python
import joblib

artifacts = joblib.load("irrigation_app.pkl")
model      = artifacts["model"]
encoders   = artifacts["encoders"]
features   = artifacts["selected_features"]
target_map = artifacts["target_map"]
```

**Preprocessing new data before inference:**
```python
def create_features(df):
    df["ET_Proxy"] = (df["Temperature_C"] * df["Sunlight_Hours"] * df["Wind_Speed_kmh"]) / (df["Humidity"] + 1)
    df["Water_Stress_Index"] = df["ET_Proxy"] / (df["Soil_Moisture"] + 1)
    df["Rain_Temp_Ratio"] = df["Rainfall_mm"] / (df["Temperature_C"] + 1)
    return df
```

---

## Tech Stack

| Library | Purpose |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `matplotlib`, `seaborn` | Visualisation |
| `scikit-learn` | Preprocessing, cross-validation, Random Forest, metrics |
| `lightgbm` | LightGBM classifier |
| `xgboost` | XGBoost classifier |
| `catboost` | CatBoost classifier |
| `scipy` | ANOVA F-tests for feature selection |
| `joblib` | Model serialisation |

---

## How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/cipherX2433/irrigation-need-prediction.git
   cd irrigation-need-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn lightgbm xgboost catboost scipy joblib
   ```

3. **Place the dataset**  
   Put `train.csv` and `test.csv` inside a `dataset/` folder (or update the paths in the notebook).

4. **Run the notebook**  
   Open `Irrigation_need_analysis.ipynb` in Jupyter or Google Colab and run all cells top-to-bottom.

5. **Load the saved model for inference**
   ```python
   import joblib, pandas as pd

   artifacts = joblib.load("saved/irrigation_app.pkl")
   model     = artifacts["model"]
   encoders  = artifacts["encoders"]
   features  = artifacts["selected_features"]

   # Preprocess your new data, then:
   predictions = model.predict(new_data[features])
   ```
