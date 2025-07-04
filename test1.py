# SEIS-763 Group Project include group member names with an introduction explaining what problem we are solving (NEEDS REVISION)
# - We are a state program that provides low interest loans to low income family.  
# - The maximum amount that we are allowed to approve on a loan is $300,000
# - Default rates tend to be high and we have a limited budget, therefore our goal is to maximize the amount allocated to our agency by predicting home values based on a multitude of different predictors in order to best mitigate loss if a home goes into default and maximize purchasing power to help as many families as possible.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report

import pandas as pd
import numpy as np

df = pd.read_csv('train.csv')

num_rows = df.shape
print(f"Number of rows in the dataset: {num_rows}")
print(df.head())

"""### Drop any rows that are missing a 'SalePrice' value prior to classification (David-Start)"""

df = df.dropna(subset=['SalePrice'])
print(df.shape)

"""### Create a Histogram and a Boxplot of the SalePrice using the dataset to determine the price range of our classes.  """

# Plot a histogram of SalePrice
plt.figure(figsize=(10, 5))
plt.hist(df['SalePrice'], bins=50, edgecolor='black')
plt.title('Histogram of SalePrice')
plt.xlabel('SalePrice')
plt.ylabel('Number of Homes')
plt.grid(axis='y', alpha=0.5)
plt.show()

# Plot a boxplot of SalePrice
plt.figure(figsize=(10, 2))
plt.boxplot(df['SalePrice'], vert=False, patch_artist=True)
plt.title('Boxplot of SalePrice')
plt.xlabel('SalePrice')
plt.show()

"""### Remove Outliers on our Target (SalePrice) Based on Low Interest Government Financing to Low Income Families"""

outliers = {}
q1 = df['SalePrice'].quantile(0.25)
q3 = df['SalePrice'].quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

# Set upper limit to $300,000 due to Organizational Business Rules
mortgage_upper = np.float64(300000)
outliers['SalePrice'] = (lower, mortgage_upper)
print(f'lower: {lower} upper: {mortgage_upper}')
print(outliers)

"""## Use Classification on the SalePrice column (target) over predicting SalePrice Directly we have determined provides the following:

### Risk Management:
- Price bins clearly correlate to risk and affordability, helping loan officers make better lending decisions.
- Predicting a range is easier and more actionable than pinpointing exact home price, especially given limited data (1,500 records).

### Interpretability:
- Price categories are easily explained to non-technical stakeholders (loan officers, regulators, buyers).

### Robustness and Stability
- Predicting precise numeric prices is harder due to high variability and data noise.
- Classification stabilizes predictions into clearly actionable bins.
"""

# Step 1: Create meaningful, fixed bins
bins = [0, 100000, 150000, 200000, 250000, 300000]
labels = [
    'Under $100K',
    '$100K–$150K',
    '$150K–$200K',
    '$200K–$250K',
    '$250K–$300K'
]

df['PriceCategory'] = pd.cut(df['SalePrice'], bins=bins, labels=labels, include_lowest=True)

# Step 2: Filter out any rows where binning failed (NaN)
df = df[df['PriceCategory'].notna()]

# Step 3: One-hot encode other categorical variables, but NOT PriceCategory as it's the target
# Identify categorical columns to encode (excluding PriceCategory)
categorical_cols_to_encode = df.select_dtypes(include=['object', 'category']).columns.tolist()
if 'PriceCategory' in categorical_cols_to_encode:
    categorical_cols_to_encode.remove('PriceCategory')

# Apply one-hot encoding to the selected categorical columns
df = pd.get_dummies(df, columns=categorical_cols_to_encode, prefix='PC')

# Quickly check the distribution of the target variable BEFORE one-hot encoding it
print(df['PriceCategory'].value_counts().sort_index())

import pandas as pd
import matplotlib.pyplot as plt
bin_counts = df['PriceCategory'].value_counts().sort_index()

plt.figure(figsize=(8, 5))
bin_counts.plot(kind='barh')  # <-- horizontal bars
plt.ylabel('Price Bin')       # Bin labels on Y axis
plt.xlabel('Number of Records')
plt.title('Number of Records per Price Bin')
plt.tight_layout()
plt.show()

"""## Finished With Classifying the Target"""

# Drop dupilicate data
df = df.drop_duplicates()
print ("After dropping duplicates: The number of rows: ", df.shape)

# Fill in missing data for numerical columns with 0
numerical_cols = df.select_dtypes(include=np.number).columns
df[numerical_cols] = df[numerical_cols].fillna(0)

# Verify the missing data are gone
print(df.isnull().sum())

# Find the missing data
total_missing = df.isnull().sum().sum()
print(total_missing)

numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
print(numerical_columns)

import scipy.stats as stats

df[numerical_columns] = df[numerical_columns].apply(stats.zscore)

other_columns = df.select_dtypes(include=['object']).columns.tolist()
print(other_columns)

df = pd.get_dummies(df, columns=other_columns, drop_first=True)
print(df.shape)
print(df.head())

print(df.isnull().sum().sum())  # Total missing values
print(df.isnull().sum())        # Missing values per column

# Create a copy of the DataFrame to work with for filtering and subsequent steps.
# The outlier filtering based on the 'outliers' dictionary is removed here
# as it was causing the DataFrame to become empty when applied after standardization.
# Outlier handling should be considered carefully in the context of the classification task
# and applied appropriately, possibly during preprocessing or as part of model training.
df_filtered = df.copy()

# Ensure PriceCategory is present for subsequent steps, assuming it was created earlier
if 'PriceCategory' not in df_filtered.columns:
    print("Warning: 'PriceCategory' column not found. Please ensure it was created in a previous step.")
    # If PriceCategory is critical and missing, you might need to recreate it or stop.
    # For now, assuming it was created and might have been dropped by get_dummies if not handled.
    # If get_dummies was applied to PriceCategory, you should work with the dummy columns instead.
    # Based on the notebook structure, PriceCategory is used as the target 'y', so it should NOT be one-hot encoded at this stage.
    # I will revert the one-hot encoding of PriceCategory from cell HY8rdCVXVYZV if it was applied there prematurely.
    # Looking back at HY8rdCVXVYZV, it *does* one-hot encode PriceCategory. This is incorrect for using PriceCategory as the target 'y'.
    # I will modify HY8rdCVXVYZV to *not* one-hot encode PriceCategory.

# Reverting premature one-hot encoding of PriceCategory from cell HY8rdCVXVYZV
# Check if the dummy columns for PriceCategory exist and revert if necessary
pc_dummy_cols = [col for col in df_filtered.columns if col.startswith('PC_')]
if pc_dummy_cols:
    # Assuming the original PriceCategory column might have been dropped, try to infer it
    # This is a complex situation due to the notebook state.
    # A better approach is to fix the one-hot encoding in HY8rdCVXVYZV directly.
    print("Attempting to revert PriceCategory one-hot encoding if it occurred.")
    # I will rely on fixing HY8rdCVXVYZV to not one-hot encode PriceCategory.
    # For now, just ensure df_filtered is a copy and the problematic filtering is removed.
    pass # The fix is primarily removing the filtering line.


# This print was redundant with the cell above
# print(df.shape)

print(df_filtered.shape)


# Theisnull().sum().sum() and isnull().sum() checks are useful but can be done later
# after more comprehensive missing value handling.
# print(df_filtered.isnull().sum().sum())  # Total missing values
# print(df_filtered.isnull().sum())        # Missing values per column

"""## Splitting Predictors, Dropping Highly Correlated Columns, and Standardizing"""

# Split predictors and target
X = df_filtered.drop(columns=["SalePrice", "PriceCategory"])
y = df_filtered["PriceCategory"]

print("Shape BEFORE dropping highly correlated columns:", X.shape)

# Compute correlation matrix
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Identify columns to drop
to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]

print("\nHighly correlated columns to drop:")
for col in to_drop:
    print("-", col)

# Drop the columns
X_reduced = X.drop(columns=to_drop)

print("Shape AFTER dropping highly correlated columns:", X_reduced.shape)

# Preview cleaned predictors
print("\nPreview of cleaned predictors:")
print(X_reduced.head())

# Verify again
print(df.isnull().sum().sum())  # Should be 0


# Standardize
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reduced)

# Convert back to DataFrame
X_scaled_df = pd.DataFrame(X_scaled, columns=X_reduced.columns)

# Confirm means and stds
print("\nMeans after standardization (should be ~0):")
print(X_scaled_df.mean().head())

print("\nStds after standardization (should be ~1):")
print(X_scaled_df.std().head())

print("\nFinal cleaned predictors shape:", X_scaled_df.shape)

"""**Summary of Results:**

Shape Before: (1399 rows, 189 columns)
*  Initial dataset after one-hot encoding and outlier removal.

Computed Correlation Matrix:
*  Measured pairwise relationships between predictors to detect multicollinearity.

Columns Dropped Due to High Correlation: 13
* Removed predictors with >0.85 correlation to avoid redundancy and improve model interpretability.

Shape After Dropping Columns: (1399 rows, 176 columns)
* Confirmed correct reduction in features.

Standardized All Predictors:
* Scaled features to mean ≈0 and standard deviation ≈1 to ensure comparability and support model convergence.

Means After Scaling: ~0
* Verified successful centering of each feature.

Standard Deviations After Scaling: ~1
* Verified correct scaling to unit variance.

Outcome:
* This cleaned and standardized dataset (X_scaled_df) is ready for regression modeling (Lasso, SVM, etc.) with reduced multicollinearity, balanced scales, and clear feature structure.

## Feature Selection Using:
- Lasso
- RFE

### Lasso (Initial Screening)
"""

# Step 1: Verify X_scaled_df and y
print(X_scaled_df.shape, y.shape)

# Import necessary libraries
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Lasso Feature Selection (Logistic Regression with L1 regularization)
lasso = LogisticRegression(penalty='l1', solver='saga', C=0.1, max_iter=5000, random_state=42)
lasso.fit(X_scaled_df, y)

lasso_model = SelectFromModel(lasso, prefit=True)
lasso_features = X_scaled_df.columns[lasso_model.get_support()]

"""### Visualization: Coeficients from Lasso

"""

coef_abs = np.abs(lasso.coef_).max(axis=0)
lasso_coef_df = pd.Series(coef_abs, index=X_scaled_df.columns).sort_values(ascending=False).head(20)

plt.figure(figsize=(10, 6))
lasso_coef_df.plot(kind='bar')
plt.title('Top 20 Features Selected by Lasso')
plt.ylabel('Coefficient Magnitude')
plt.tight_layout()
plt.show()

"""### RFE (Further Refinement)"""

# RFE Feature Selection
logreg = LogisticRegression(max_iter=5000, random_state=42)
rfe = RFE(estimator=logreg, n_features_to_select=20)
rfe.fit(X_scaled_df, y)

rfe_features = X_scaled_df.columns[rfe.get_support()]

"""#### Visualization: RFE Selected Features"""

# Visualize RFE rankings
rfe_ranking_df = pd.Series(rfe.ranking_, index=X_scaled_df.columns).sort_values().head(20)

plt.figure(figsize=(10, 6))
rfe_ranking_df.plot(kind='bar')
plt.title('Top 20 Features by RFE Ranking (lower is better)')
plt.ylabel('Ranking')
plt.tight_layout()
plt.show()

"""### Further Refinement using Random Forest"""

# Random Forest Feature Selection
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_scaled_df, y)

rf_importances = pd.Series(rf.feature_importances_, index=X_scaled_df.columns).sort_values(ascending=False).head(20)

"""#### Visualization: Feature importance using Random Forest"""

plt.figure(figsize=(10, 6))
rf_importances.plot(kind='bar')
plt.title('Top 20 Feature Importances from Random Forest')
plt.ylabel('Importance')
plt.tight_layout()
plt.show()

"""### Combine feature selection results"""

# Step 2: Combine features from Lasso, RFE, and Random Forest
combined_features = pd.concat([pd.Series(lasso_features), pd.Series(rfe_features), pd.Series(rf_importances.index)]).value_counts()

# Select features identified by at least two methods
final_features = combined_features[combined_features >= 2].index.tolist()

print(f'Total combined features selected: {len(final_features)}')
print('Final selected features:', final_features)

"""#### Visualization: Feature Selection with Combined Results"""

import plotly.express as px
import pandas as pd

# Convert the Series to DataFrame for Plotly
combined_features_df = combined_features.reset_index()
combined_features_df.columns = ['Feature', 'Frequency']
combined_features_df = combined_features_df.sort_values(by='Frequency', ascending=True)

# Plot using Plotly
fig = px.bar(
    combined_features_df,
    x='Frequency',
    y='Feature',
    orientation='h',
    title='Frequency of Features Selected by Lasso, RFE, and Random Forest',
    height=1600,  # Increase height for vertical scroll space
    width=900
)

fig.update_layout(
    yaxis=dict(
        tickfont=dict(size=10),
    ),
    xaxis=dict(title='Selection Frequency'),
    margin=dict(l=200, r=20, t=50, b=20)
)

fig.show()

"""### Refine DataFrame with Feature Selection"""

# Step 3: Refine dataframe to include only selected features for SVM RBF
X_final = X_scaled_df[final_features]

"""### Using SVM and RBF ### """

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Step 7 Applying SVM RBF
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42, stratify=y)

svm_rbf = SVC(kernel='rbf', random_state=42)
svm_rbf.fit(X_train, y_train)

print("Step 7 complete: SVM with RBF kernel has been trained.")

y_pred = svm_rbf.predict(X_test)
print("Predictions on test set:", y_pred[:10])  # Shows first 10 predictions

print("Classes recognized by the model:", svm_rbf.classes_)

from sklearn.metrics import classification_report, confusion_matrix

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))