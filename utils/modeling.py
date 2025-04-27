import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def train_model(X_train_scaled, y_train):
    """Train a Random Forest model."""
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    return rf

def evaluate_model(model, X_test_scaled, y_test):
    """Evaluate the model and return metrics."""
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    return mse, rmse, r2, y_pred

def predict_single(model, scaler, input_data, feature_names):
    """Make a prediction for a single input."""
    input_df = pd.DataFrame([input_data], columns=feature_names)
    input_scaled = scaler.transform(input_df)
    return model.predict(input_scaled)[0]