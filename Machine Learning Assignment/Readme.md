# Machine Learning Assignment - NOAA Hurricane & Typhoon Classification

## Project Overview

This project develops and evaluates machine learning models to classify tropical cyclones from the NOAA (National Hurricane Center) historical database. The goal is to accurately predict storm status categories based on meteorological features, providing insights into storm classification patterns and the impact of data imbalance on model performance.

**Author:** Yash Bishnoi  
**Student ID:** 2665498  
**Dataset:** NOAA Hurricanes and Typhoons, 1851-2014

---

## Dataset Description

### Source and Context
The National Hurricane Center (NHC) publishes tropical cyclone historical data in HURDAT format (HURricane DATabase). This dataset contains six-hourly information on location, maximum winds, central pressure, and storm size for all known tropical and subtropical cyclones in the Atlantic and North Pacific basins.

### Storm Status Categories (Target Variable)
The dataset classifies storms into 12 categories:

| Code | Description | Intensity |
|------|-------------|-----------|
| TD | Tropical Depression | < 34 knots |
| TS | Tropical Storm | 34-63 knots |
| HU | Hurricane | > 64 knots |
| EX | Extratropical Cyclone | Any intensity |
| SD | Subtropical Depression | < 34 knots |
| SS | Subtropical Storm | > 34 knots |
| LO | Low (non-tropical) | Any intensity |
| WV | Tropical Wave | Any intensity |
| DB | Disturbance | Any intensity |
| ET | Extratropical Transition | Any intensity |
| PT | Post-Tropical | Any intensity |
| ST | Subtropical Transition | Any intensity |

### Features Used

**Geographic Features:**
- **Latitude**: Storm location (degrees, converted from "28.0N" format)
- **Longitude**: Storm location (degrees, converted from "90.5W" format)

**Meteorological Features:**
- **Maximum Wind**: Maximum sustained wind speed (knots)
- **Minimum Pressure**: Central minimum pressure (mb)
- **Wind Quadrants**: High/Moderate/Low wind speeds in NE, SE, SW, NW directions

**Engineered Features:**
- **Month_Sin/Month_Cos**: Cyclical encoding of seasonality
- **Year**: Extracted for temporal analysis

### Initial Data Quality Assessment

The raw dataset revealed significant data quality challenges:
- **Missing Data**: High presence of missing values in 'Event' column, wind radii columns, and 'Minimum Pressure'
- **Placeholder Values**: Common NOAA placeholders (-999, -9999, 0) used for missing observations
- **Unnamed Storms**: Records with 'UNNAMED' storm names required filtering

---

## Data Cleaning & Feature Engineering

### 1. Coordinate Cleaning
Converted string-based coordinates to numeric values:
```python
# "28.0N" → 28.0 (positive)
# "90.5W" → -90.5 (negative)
```

### 2. Data Filtering
- Removed 6,000+ 'UNNAMED' storm records
- Final dataset: **46,499 records, 25 features**

### 3. Placeholder Handling
Replaced placeholder values with NaN for proper statistical treatment:
```python
cols_to_clean = ['Maximum Wind', 'Minimum Pressure', 'Low Wind NE', ...]
df[col] = df[col].replace([-999, -9999, 0], np.nan)
```

### 4. Imputation Strategy
- **Minimum Pressure**: Filled with median per storm status category
- **Wind Radii**: Filled remaining NaNs with 0 (assuming no recorded wind = 0)

### 5. Date Feature Engineering
```python
# Convert to datetime and extract month
df['Date'] = pd.to_datetime(df['Date'], format='mixed')
df['Month'] = df['Date'].dt.month

# Cyclical encoding for seasonality
df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
```

---

## Methodology

### Data Preparation Pipeline

1. **Feature Selection**: Selected 18 numerical features (Latitude, Longitude, Maximum Wind, Minimum Pressure, Month_Sin, Month_Cos, and 12 wind quadrant features)

2. **Target Encoding**: Used LabelEncoder to convert categorical Status labels to integers (0-11)

3. **Stratified Splitting**: 
   - Training: 70% (32,549 samples)
   - Validation: 15% (6,975 samples)
   - Test: 15% (6,975 samples)

4. **Feature Scaling**: StandardScaler (mean=0, std=1) fitted only on training data

### Model Approaches

#### Model 1: Random Forest (10-Fold Cross-Validation)
```python
RandomForestClassifier(
    n_estimators=500,
    criterion='entropy',
    min_samples_split=5,
    class_weight='balanced',
    random_state=42
)
```

**Why This Approach:**
- Robust to overfitting
- Provides feature importance rankings
- Handles imbalanced data via class_weight
- 500 estimators for stable predictions

#### Model 2: XGBoost with GridSearchCV
```python
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [4, 6, 8],
    'n_estimators': [200, 500]
}
```

**Why This Approach:**
- Excellent gradient boosting performance
- GridSearchCV for systematic hyperparameter tuning
- Built-in regularization (reg_alpha, reg_lambda)
- Best parameters: learning_rate=0.1, max_depth=8, n_estimators=500

#### Model 3: Deep Learning Neural Network
```python
# Architecture: Input(18) → Dense(256) → Dense(128) → Dense(64) → Output(12)
# Regularization: L2(0.001), BatchNorm, Dropout(0.3/0.2)
# Optimizer: Adam with EarlyStopping & ReduceLROnPlateau
```

**Why This Approach:**
- Explore non-linear feature relationships
- Test deep learning on tabular data
- Multiple regularization techniques for generalization

#### Model 4: SMOTE-Enhanced Random Forest
```python
smote = SMOTE(random_state=42, k_neighbors=3)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
```

**Why This Approach:**
- Address class imbalance through synthetic oversampling
- Generate new minority class samples (not just duplication)
- Improve recall for rare storm categories

---

## Results & Model Comparison

### Performance Summary

| Model | Accuracy | Macro F1 | Best For |
|-------|----------|---------|----------|
| Random Forest (10-Fold) | 95.35% ± 0.27% | ~0.71 | Stable baseline |
| XGBoost (Optimized) | 94.80% | ~0.66 | Speed & efficiency |
| Deep Learning | 91.03% ± 0.14% | ~0.63 | Pattern learning |
| **SMOTE-Enhanced RF** | **95.40%** | **0.85** | **Balanced performance** |

### Feature Importance (Random Forest)

The most influential features for storm classification:

1. **Minimum Pressure** - Primary discriminator
2. **Maximum Wind** - Strong predictive power
3. **Latitude** - Geographic distribution patterns
4. **Longitude** - Regional storm characteristics
5. **Month_Sin/Month_Cos** - Captures seasonality

### Class-Specific Performance (Key Finding)

**Common Classes (High Support):**
- Hurricane (HU): 100% precision/recall
- Tropical Storm (TS): 98-99% precision/recall
- Tropical Depression (TD): 91-96% precision/recall

**Rare Classes (Low Support):**
- Pre-SMOTE: Often 0-70% recall for classes like PT, ST, SD
- Post-SMOTE: Significant improvements (SD recall: 47% → 65%)

### SMOTE Impact Analysis

**Before SMOTE (XGBoost):**
```
              precision    recall  f1-score
DB               0.78      0.73      0.76
ET               0.78      0.70      0.74
LO               0.78      0.68      0.72
SD               0.61      0.47      0.53
SS               0.83      0.55      0.66
WV               0.72      0.65      0.68
```

**After SMOTE (Random Forest):**
```
              precision    recall  f1-score
DB               0.80      0.77      0.78 (+2%)
ET               0.62      0.83      0.71 (+13% recall)
LO               0.77      0.73      0.75 (+5%)
SD               0.61      0.65      0.63 (+12% recall)
SS               0.71      0.73      0.72 (+7% recall)
WV               0.80      0.70      0.74 (+6%)
```

**Macro F1-Score Improvement:** 0.66 → 0.85 (29% relative improvement)

---

## Key Insights

### 1. Meteorological Validation
The models confirm known meteorological relationships:
- **Strong negative correlation** between Maximum Wind and Minimum Pressure (r ≈ -0.87)
- Clear class separation for HU/TS/TD based on wind thresholds
- Distinct geographic patterns for different storm types

### 2. Data Imbalance Challenge
- 12 classes with highly skewed distribution
- TS (20,000+) vs ST/PT (<10 samples)
- Standard metrics (accuracy) mask poor minority class performance

### 3. Model Selection Insights
- **Tree-based models outperformed Deep Learning** on this tabular dataset
- Ensemble methods (RF, XGBoost) more effective for clear feature boundaries
- DNN requires extensive tuning for tabular data

### 4. SMOTE Effectiveness
- Maintained overall accuracy (95.4%)
- Dramatically improved minority class detection
- Essential for real-world applications requiring rare event identification

---

## Conclusion

This project successfully developed and compared multiple machine learning approaches for tropical cyclone classification. The key findings demonstrate that:

1. **Ensemble tree-based models** (Random Forest, XGBoost) are highly effective for meteorological classification tasks, achieving >94% accuracy on this tabular dataset.

2. **Addressing data imbalance is critical** - while overall accuracy metrics appear strong, the SMOTE-enhanced Random Forest revealed that minority classes were significantly misclassified without proper handling.

3. **The SMOTE-enhanced Random Forest emerges as the optimal choice**, offering:
   - High overall accuracy (95.4%)
   - Balanced performance across all 12 storm categories
   - Macro F1-score improvement of 29% over baseline models
   - Practical reliability for operational storm classification

4. **Feature engineering proved valuable** - cyclical month encoding captured seasonal patterns, and geographic coordinates provided strong discriminative power.

### Recommendations for Future Work
- Collect more data for extremely rare storm categories
- Explore time-series models for storm track prediction
- Implement ensemble of best-performing models
- Consider additional features (sea surface temperature, humidity indices)

---

## Dependencies

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
tensorflow
xgboost
imbalanced-learn
```

---

## Files
- **Notebook:** `Machine_Learning_Assignment_Yash_Bishnoi_2665498.ipynb`
- **Dataset:** `hurricane.csv` (NOAA HURDAT2 format)
- **Documentation:** `README.md`

---

*Generated for Machine Learning Assignment - University of Dundee*

