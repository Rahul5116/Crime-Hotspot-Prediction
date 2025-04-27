import streamlit as st
import pandas as pd
from utils.data_processing import load_data, preprocess_data, prepare_features, scale_features
from utils.visualizations import plot_crime_distribution, plot_correlation_matrix, plot_feature_importance
from utils.modeling import train_model, evaluate_model, predict_single
from sklearn.model_selection import train_test_split

# Set page configuration
st.set_page_config(page_title="Crime Data Analysis App", layout="wide")

# Title and description
st.title("Crime Data Analysis and Prediction")
st.write("Upload crime_data.csv to explore the data, visualize patterns, and predict Total Cognizable IPC Crimes using a Random Forest model.")

# File uploader
uploaded_file = st.file_uploader("Upload crime_data.csv", type="csv")

if uploaded_file is not None:
    # Load and preprocess data
    data = load_data(uploaded_file)
    data, le_state, le_district = preprocess_data(data)
    st.success("Data loaded successfully!")
    st.write("### Data Preview")
    st.dataframe(data.head())

    # EDA Section
    st.write("### Exploratory Data Analysis")

    # Data summary
    st.write("#### Data Summary")
    st.write(f"**Shape**: {data.shape}")
    st.write(f"**Missing Values**: {data.isnull().sum().sum()}")
    st.write("**Descriptive Statistics**")
    st.dataframe(data.describe())

    # Visualizations
    st.write("#### Visualizations")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Distribution of Total Cognizable IPC Crimes**")
        fig = plot_crime_distribution(data)
        st.pyplot(fig)

    with col2:
        st.write("**Correlation Matrix**")
        fig = plot_correlation_matrix(data)
        st.pyplot(fig)

    # Model Training Section
    st.write("### Model Training and Evaluation")

    # Prepare features and target
    X, y = prepare_features(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Train model
    rf = train_model(X_train_scaled, y_train)

    # Evaluate model
    mse, rmse, r2, y_pred = evaluate_model(rf, X_test_scaled, y_test)

    st.write("#### Random Forest Model Performance")
    st.write(f"**Mean Squared Error (MSE)**: {mse:.2f}")
    st.write(f"**Root Mean Squared Error (RMSE)**: {rmse:.2f}")
    st.write(f"**R² Score**: {r2:.2f}")

    # Feature importance
    st.write("#### Feature Importance")
    fig = plot_feature_importance(rf.feature_importances_, X.columns)
    st.pyplot(fig)

    # Prediction Interface
    st.write("### Predict Total Cognizable IPC Crimes")

    # Get top 5 features
    feature_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)[:5]

    # Create input fields for key features
    st.write("Enter values for the top features:")
    input_data = {}
    for feature in feature_importance.index:
        input_data[feature] = st.number_input(f"{feature}", min_value=0.0, value=0.0)

    # Add encoded features (State and District)
    state_options = le_state.classes_
    district_options = le_district.classes_
    input_data['State_Encoded'] = st.selectbox("Select State", options=range(len(state_options)), format_func=lambda x: state_options[x])
    input_data['District_Encoded'] = st.selectbox("Select District", options=range(len(district_options)), format_func=lambda x: district_options[x])

    # Add remaining features with default 0
    for feature in X.columns:
        if feature not in input_data:
            input_data[feature] = 0.0

    # Predict button
    if st.button("Predict"):
        prediction = predict_single(rf, scaler, input_data, X.columns)
        st.success(f"Predicted Total Cognizable IPC Crimes: **{prediction:.2f}**")
else:
    st.warning("Please upload a CSV file to proceed.")
    st.stop()