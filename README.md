# Walmart Sales Time Series Forecasting Using Machine and Deep Learning
 Time Series Forecasting of Walmart Sales Data using Deep Learning and Machine Learning


# Walmart Recruiting - Store Sales Forecasting

Data source: [Kaggle competition – Walmart Recruiting: Store Sales Forecasting](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting)

## Datasets Used
- **train.csv**: 115,064 rows with attributes — *Store, Dept, Date, Weekly_Sales, IsHoliday*  
- **stores.csv**: 45 rows with attributes — *Store, Type, Size*  
- **features.csv**: 8,190 rows with attributes — *Store, Date, Temperature, Fuel_Price, MarkDown1–5, CPI, Unemployment, IsHoliday*  

---

## Data Preprocessing
- Missing values handled:  
  - CPI & Unemployment (585)  
  - MarkDown1 (4,158), MarkDown2 (5,269), MarkDown3 (4,577), MarkDown4 (4,726), MarkDown5 (4,140)  
  - All filled with **median values**  
- Datasets merged: `train + stores` → merged with `features` → **421,570 rows, 15 attributes**  
- Converted **Date** to datetime, set as index, split into Year, Month, Week  
- Aggregated **Weekly_Sales** with max, min, mean, median, std  
- Created **Total_MarkDown** (sum of MarkDown1–5)  
- Outliers removed with **z-score**, negative Weekly_Sales removed → final dataset: **374,247 rows, 20 columns**  
- Applied **One-Hot Encoding** to Store, Dept, Type → columns expanded to **145**  
- Normalized numerical features with **MinMaxScaler (0–1)**  

---

## Feature Selection
- Applied **Recursive Feature Elimination** with Random Forest (23 estimators)  
- Selected **24 attributes**:  
  mean, median, Week, Temperature, max, CPI, Fuel_Price, min, std, Unemployment, Month, Total_MarkDown, Dept_16, Dept_18, IsHoliday, Dept_3, Size, Dept_9, Year, Dept_11, Dept_1, Dept_5, Dept_56  

---

## Model Training
- Train/Test split: **80% / 20%**  
- Target Feature: **Weekly_Sales**  

---

## Model Results

### Linear Regression
- Accuracy: **92.28%**  
- MAE: 0.030057 | MSE: 0.0034851 | RMSE: 0.059 | R²: 0.9228  
- Params: `copy_X=True, fit_intercept=True`  

### Random Forest Regression
- Accuracy: **97.889%**  
- MAE: 0.015522 | MSE: 0.000953 | RMSE: 0.03087 | R²: 0.9788  
- Params: `n_estimators=100, criterion='mse'`  

### K Neighbors Regression
- Accuracy: **91.9726%**  
- MAE: 0.0331221 | MSE: 0.0036242 | RMSE: 0.060202 | R²: 0.9199  
- Params: `n_neighbors=1, metric='minkowski'`  

### XGBoost Regression
- Accuracy: **94.2115%**  
- MAE: 0.0267718 | MSE: 0.0026134 | RMSE: 0.051121 | R²: 0.9421  
- Params: `learning_rate=0.1, n_estimators=100, max_depth=3`  

### Custom Deep Learning Neural Network
- Accuracy: **90.5033%**  
- MAE: 0.033255 | MSE: 0.003867 | RMSE: 0.06218 | R²: 0.9144  
- Architecture: Input (23→64, relu) → Hidden (32 nodes) → Output (1 node)  
- Params: `kernel_initializer=normal, optimizer=adam, batch_size=5000, epochs=100`  

---

## Model Comparison
- **Linear Regression**: 92.28%  
- **Random Forest**: 97.889%  
- **K Neighbors**: 91.973%  
- **XGBoost**: 94.212%  
- **Deep Neural Network**: 90.503%  
