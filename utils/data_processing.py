import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(file):
    """Load the CSV file and return a DataFrame."""
    return pd.read_csv(file)

def preprocess_data(data):
    """Preprocess the data: handle missing values, encode categoricals, create new features."""
    # Handle missing values
    data = data.fillna(0)  # Replace NaN with 0 for crime counts

    # Encode categorical variables
    le_state = LabelEncoder()
    le_district = LabelEncoder()
    data['State_Encoded'] = le_state.fit_transform(data['States/UTs'])
    data['District_Encoded'] = le_district.fit_transform(data['District'])

    # Create a new feature: Total Violent Crimes
    violent_crimes = ['Murder', 'Attempt to commit Murder', 'Rape', 'Kidnapping & Abduction_Total', 'Dacoity', 'Robbery']
    data['Total_Violent_Crimes'] = data[violent_crimes].sum(axis=1)

    return data, le_state, le_district

def prepare_features(data, target='Total Cognizable IPC crimes'):
    """Prepare features and target for modeling."""
    X = data.drop(['States/UTs', 'District', 'Year', target], axis=1)
    y = data[target]
    return X, y

def scale_features(X_train, X_test):
    """Scale features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler