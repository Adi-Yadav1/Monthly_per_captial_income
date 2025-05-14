import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set the style for plots
plt.style.use('ggplot')
sns.set(style="whitegrid")

# 1. Data Collection
print("Step 1: Data Collection")
# Load the dataset
df = pd.read_csv('mpce_dataset.csv')
print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.")

# 2. Data Information and Details
print("\nStep 2: Data Information and Details")
print("\nFirst 5 rows of the dataset:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nDescriptive Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

# Check for duplicate rows
duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")

# 3. Data Preprocessing and Cleaning
print("\nStep 3: Data Preprocessing and Cleaning")

# Make a copy of the dataframe to avoid modifying the original
df_cleaned = df.copy()

# Handle missing values if any
if df_cleaned.isnull().sum().sum() > 0:
    # For numerical columns, fill with median
    num_cols = df_cleaned.select_dtypes(include=['int64', 'float64']).columns
    for col in num_cols:
        if df_cleaned[col].isnull().sum() > 0:
            df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
    
    # For categorical columns, fill with mode
    cat_cols = df_cleaned.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df_cleaned[col].isnull().sum() > 0:
            df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)

# Remove duplicate rows if any
if duplicates > 0:
    df_cleaned = df_cleaned.drop_duplicates()
    print(f"Removed {duplicates} duplicate rows.")

# Check if MPCE is calculated correctly (Total_Expenditure / Household_Size)
df_cleaned['Calculated_MPCE'] = df_cleaned['Total_Expenditure'] / df_cleaned['Household_Size']
mpce_diff = (df_cleaned['MPCE'] - df_cleaned['Calculated_MPCE']).abs()
print(f"Max difference between provided MPCE and calculated MPCE: {mpce_diff.max()}")
print(f"Mean difference between provided MPCE and calculated MPCE: {mpce_diff.mean()}")

# If the difference is negligible, we can use the provided MPCE
# Otherwise, we might want to recalculate it
if mpce_diff.mean() < 0.01:
    print("MPCE values are correctly calculated.")
else:
    print("There might be discrepancies in MPCE calculation.")

# Drop the calculated MPCE column as we'll use the original
df_cleaned.drop('Calculated_MPCE', axis=1, inplace=True)

# 4. Exploratory Data Analysis (EDA)
print("\nStep 4: Exploratory Data Analysis")

# Create a directory for saving plots
import os
if not os.path.exists('plots'):
    os.makedirs('plots')

# EDA 1: Distribution of MPCE
plt.figure(figsize=(10, 6))
sns.histplot(df_cleaned['MPCE'], kde=True)
plt.title('Distribution of Monthly Per Capita Expenditure (MPCE)')
plt.xlabel('MPCE')
plt.ylabel('Frequency')
plt.savefig('plots/mpce_distribution.png')
plt.close()

# EDA 2: MPCE by State
plt.figure(figsize=(12, 8))
sns.boxplot(x='State', y='MPCE', data=df_cleaned)
plt.title('MPCE Distribution by State')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plots/mpce_by_state.png')
plt.close()

# EDA 3: MPCE by Rural/Urban
plt.figure(figsize=(10, 6))
sns.boxplot(x='Rural_Urban', y='MPCE', data=df_cleaned)
plt.title('MPCE Distribution by Rural/Urban')
plt.savefig('plots/mpce_by_rural_urban.png')
plt.close()

# EDA 4: Correlation between MPCE and other numerical variables
numerical_cols = df_cleaned.select_dtypes(include=['int64', 'float64']).columns
correlation = df_cleaned[numerical_cols].corr()
plt.figure(figsize=(14, 10))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numerical Variables')
plt.tight_layout()
plt.savefig('plots/correlation_matrix.png')
plt.close()

# EDA 5: MPCE by Education Level
plt.figure(figsize=(12, 8))
sns.boxplot(x='Education_Level', y='MPCE', data=df_cleaned)
plt.title('MPCE Distribution by Education Level')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plots/mpce_by_education.png')
plt.close()

# EDA 6: MPCE by Employment Status
plt.figure(figsize=(12, 8))
sns.boxplot(x='Employment_Status', y='MPCE', data=df_cleaned)
plt.title('MPCE Distribution by Employment Status')
plt.tight_layout()
plt.savefig('plots/mpce_by_employment.png')
plt.close()

# EDA 7: Scatter plot of MPCE vs Monthly Income
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Monthly_Income', y='MPCE', data=df_cleaned)
plt.title('MPCE vs Monthly Income')
plt.savefig('plots/mpce_vs_income.png')
plt.close()

# 5. Feature Engineering and Selection
print("\nStep 5: Feature Engineering and Selection")

# Drop the Household_ID column as it's just an identifier
df_cleaned.drop('Household_ID', axis=1, inplace=True)

# Define features and target variable
X = df_cleaned.drop('MPCE', axis=1)
y = df_cleaned['MPCE']

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Print the columns
print(f"Categorical columns: {categorical_cols}")
print(f"Numerical columns: {numerical_cols}")

# Create preprocessing pipelines for both numerical and categorical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Model Selection and Training
print("\nStep 6: Model Selection and Training")

# Define the models to evaluate
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'KNN Regressor': KNeighborsRegressor(),
    'Support Vector Regressor': SVR(),
    'XGBoost': XGBRegressor(random_state=42)
}

# Dictionary to store model results
model_results = {}

# Train and evaluate each model
for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Create a pipeline with preprocessing and the model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Store the results
    model_results[name] = {
        'pipeline': pipeline,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
    
    print(f"{name} - MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")

# 7. Model Comparison and Selection
print("\nStep 7: Model Comparison and Selection")

# Compare models based on R² score
r2_scores = {name: results['r2'] for name, results in model_results.items()}
best_model_name = max(r2_scores, key=r2_scores.get)
best_model = model_results[best_model_name]['pipeline']

print(f"\nBest model based on R² score: {best_model_name} with R² = {r2_scores[best_model_name]:.4f}")

# Create a bar chart to compare R² scores
plt.figure(figsize=(12, 6))
models_names = list(r2_scores.keys())
r2_values = list(r2_scores.values())
bars = plt.bar(models_names, r2_values, color='skyblue')
plt.title('Model Comparison - R² Score')
plt.xlabel('Models')
plt.ylabel('R² Score')
plt.xticks(rotation=45)
plt.tight_layout()

# Highlight the best model
best_index = models_names.index(best_model_name)
bars[best_index].set_color('green')

plt.savefig('plots/model_comparison.png')
plt.close()

# 8. Save the best model
print("\nStep 8: Saving the Best Model")
with open('best_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)
print(f"Best model ({best_model_name}) saved as 'best_model.pkl'")

print("\nAnalysis and model training completed successfully!")