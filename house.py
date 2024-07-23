import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load dataset
data_path = "C://hpp//Bengaluru_House_Data (1).csv"
data = pd.read_csv(data_path)

# Drop irrelevant columns
data = data.drop(['availability', 'society'], axis=1)

# Clean data functions
def convert_size(size):
    try:
        return int(size.split()[0])
    except Exception as e:
        return np.nan

def convert_total_sqft(total_sqft):
    try:
        if '-' in str(total_sqft):
            sqft_values = [float(x) for x in total_sqft.split('-')]
            return sum(sqft_values) / len(sqft_values)
        else:
            return float(total_sqft)
    except Exception as e:
        return np.nan

data['size'] = data['size'].apply(convert_size)
data['total_sqft'] = data['total_sqft'].apply(convert_total_sqft)

# Drop rows with missing values
data = data.dropna()

# Prepare features and target variable
X = data.drop(['price'], axis=1)
y = data['price']

# Handle categorical features: area_type and location
X = pd.get_dummies(X, columns=['area_type', 'location'], drop_first=True)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model - using Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f'RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}')
print(f'R2 Score: {r2_score(y_test, y_pred)}')

# Save the model and the columns
joblib.dump(model, 'house_price_model_rf.pkl')
joblib.dump(X.columns, 'model_columns_rf.pkl')
