# Bangalore Home Prices Prediction

This project is an end-to-end machine learning pipeline that predicts residential property prices in Bengaluru, India. It walks through every stage of a real-world ML workflow—from raw data ingestion to a serialized, production-ready model—and explains both fundamental and advanced concepts so that beginners can follow along.

## Dataset
The project uses the **Bengaluru House Price** dataset (`bengaluru_house_prices.csv`), which contains features such as area type, location, society, total square footage, number of bathrooms, balconies, bedrooms (BHK), and price (in lakhs).

## Notebook: `bengaluru_house_prices_prediction.ipynb`

### Topics Covered

1. **Imports and Data Load**
   - Loading libraries: Pandas, NumPy, Seaborn, Matplotlib, Scikit-learn, Plotly, SciPy
   - Reading the CSV dataset into a DataFrame

2. **Interactive Real Estate Analytics (EDA)**
   - Interactive visualisations built with Plotly
   - **Data Distribution** – price histograms, box plots, violin plots across BHK segments
   - **Data Composition** – area-type pie/donut chart, top-15 location treemap
   - **Data Relationship** – scatter plots (sqft vs price by area type & BHK), Pearson correlation heatmap
   - **Data Comparison** – median price bar charts by area type and by top locations
   - Univariate, bivariate, and multivariate analysis

3. **Data Preprocessing**
   - Dropping irrelevant columns (`availability`)
   - Identifying and handling missing values
   - Group-based mode imputation for the `society` column
   - Zero-fill for the `balcony` column; row-dropping for `bath`, `location`, and `size`
   - Standardising text: stripping whitespace and fixing inconsistent category names in `area_type`, `location`, and `society`

4. **Data Exploration**
   - Examining `area_type` value counts
   - Parsing the `size` column into a numeric `BHK` feature
   - Converting `total_sqft` from mixed formats (plain numbers and ranges) to a single numeric value

5. **Feature Engineering**
   - Deriving `price_per_sqft` (price × 100,000 / total_sqft)
   - Grouping sparse locations (≤ 10 data points) into an `other` category to reduce cardinality

6. **Outlier Removal**
   - Removing listings with unrealistically small per-room area (< 250 sqft per BHK)
   - Location-wise mean ± 2 std removal on `price_per_sqft`
   - Cross-BHK price consistency check (e.g. a 2 BHK must not be cheaper than the mean 1 BHK price in the same locality)
   - Removing rows where the number of bathrooms exceeds BHK + 2

7. **Encoding**
   - One-hot encoding of `location`, `society`, and `area_type` using `pd.get_dummies`
   - Dropping original categorical columns after encoding

8. **Model Training and Selection**
   - Train/test split (80/20)
   - Candidates evaluated: Linear Regression, Ridge, Lasso, Decision Tree, Gradient Boosting
   - Hyperparameter tuning with `GridSearchCV` and `ShuffleSplit` cross-validation
   - **Winner: Gradient Boosting Regressor** (`n_estimators=200`, `learning_rate=0.1`, `max_depth=3`)
   - Manual prediction testing on sample listings with known prices

9. **Model Export**
   - Serialising the trained model to `banglore_home_price_prediction_model.pkl` with `pickle`
   - Exporting column metadata (location, society, area-type, and other feature columns) to a JSON file for use in a prediction API

## Scope

| Stage | Status |
|---|---|
| Exploratory Data Analysis | ✅ |
| Data Cleaning & Preprocessing | ✅ |
| Feature Engineering | ✅ |
| Outlier Detection & Removal | ✅ |
| Categorical Encoding | ✅ |
| Model Selection & Hyperparameter Tuning | ✅ |
| Model Serialisation | ✅ |

The notebook does **not** include a deployment layer (e.g. a Flask/FastAPI server or front-end UI), but the exported `.pkl` and column-metadata JSON provide everything needed to build one.
