# Crime Hotspot Prediction 🚨🔍

Welcome to the **Crime Hotspot Prediction** web app built with **Streamlit**, providing an interactive platform to analyze crime data in India (2014) and predict the total cognizable IPC crimes using a **Random Forest** model. This project allows users to explore crime data, visualize trends, and make predictions based on crime statistics for different districts.

## 🎯 Features

- **Data Upload**: Upload your crime data (CSV format) to explore and predict crime trends.
- **Data Exploration**: View data summaries, histograms, and correlation heatmaps.
- **Prediction**: Predict the total cognizable IPC crimes for any district based on input features using a trained Random Forest model.
- **Model Evaluation**: The app evaluates the model's performance with MSE, RMSE, and R² scores, and visualizes feature importance.
- **Interactive UI**: User-friendly interface built using **Streamlit** for easy navigation and interaction.

## 📂 Directory Structure

```
crime_hotspot_prediction/
│
├── app.py                  # Main Streamlit app script
├── README.md               # Project documentation (this file)
├── requirements.txt        # List of dependencies
├── data/
│   └── crime_data.csv      # Input dataset (user-provided)
├── utils/
│   ├── data_processing.py  # Data loading and preprocessing
│   ├── modeling.py         # Model training and evaluation
│   └── visualizations.py   # Visualization functions
```

### `data/crime_data.csv`:
Ensure your dataset contains the following columns for accurate analysis:
- `States/UTs`
- `District`
- `Year`
- `Murder`
- `Rape`
- `Total Cognizable IPC crimes`
  
The dataset should be in CSV format for seamless integration.

### `utils/`:
Contains modular scripts to handle data preprocessing, machine learning model training, and visualizations.

## 🛠️ Installation

Follow these steps to set up the project locally. Python 3.11 is recommended as Python 3.13 may have compatibility issues with some dependencies.

### Prerequisites

- **Python 3.11**: Download from [python.org](https://www.python.org/downloads/) (Ensure that "Add Python to PATH" is checked during installation).
- **Git**: Make sure **Git** is installed to clone the repository.
- **Virtual Environment**: It's a good practice to use a virtual environment to manage dependencies.

### Step-by-Step Setup

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/rahul/crime-hotspot-prediction.git
    cd crime-hotspot-prediction
    ```

2. **Create a Virtual Environment**:
    ```bash
    python -m venv venv
    .\venv\Scripts\activate  # On Windows
    # source venv/bin/activate  # On macOS/Linux
    ```

3. **Install Dependencies**:
    Create and save the following `requirements.txt` in the root directory:
    
    ```txt
    streamlit>=1.29.0
    pandas>=2.1.4
    numpy>=1.26.4
    matplotlib>=3.8.2
    seaborn>=0.13.0
    scikit-learn>=1.3.2
    ```

    Install the dependencies with:
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

4. **Prepare the Dataset**:
    - Place `crime_data.csv` inside the `data/` directory.
    - Make sure the dataset has the required columns (`States/UTs`, `District`, `Year`, `Total Cognizable IPC crimes`).

5. **Run the App**:
    Start the Streamlit app by running the following command:
    ```bash
    streamlit run app.py
    ```
    The app will open in your browser at [http://localhost:8501](http://localhost:8501).

## 🚀 Usage

### 1. Upload Data:
- Click on the file uploader widget in the Streamlit app.
- Upload your `crime_data.csv` file to begin analyzing the data.

### 2. Data Exploration:
- **Data Summary**: View basic statistics, shape of the data, and missing values.
- **Visualizations**: See visual representations of the total cognizable IPC crimes distribution and the correlation heatmap.

### 3. Train the Model:
- The app will automatically train a Random Forest model on the dataset.
- It will evaluate the model’s performance with metrics such as **MSE**, **RMSE**, and **R²**.

### 4. Make Predictions:
- Enter values for key features such as `Murder`, `Rape`, `Robbery`, etc.
- Select the **State** and **District** from dropdown lists.
- Press **Predict** to view the predicted total cognizable IPC crimes.

## 💻 Code Walkthrough

### 1. `requirements.txt`

This file lists the necessary dependencies for the project:

```txt
streamlit>=1.29.0
pandas>=2.1.4
numpy>=1.26.4
matplotlib>=3.8.2
seaborn>=0.13.0
scikit-learn>=1.3.2
```

### 2. `app.py`

The main Streamlit app script controls the user interface and workflow:

```python
import streamlit as st
import pandas as pd
from utils.data_processing import load_data, preprocess_data, prepare_features, scale_features
from utils.visualizations import plot_crime_distribution, plot_correlation_matrix, plot_feature_importance
from utils.modeling import train_model, evaluate_model, predict_single
from sklearn.model_selection import train_test_split

# Streamlit configuration
st.set_page_config(page_title="Crime Data Analysis App", layout="wide")

# Title and description
st.title("Crime Data Analysis and Prediction")
st.write("Upload crime_data.csv to explore the data, visualize patterns, and predict Total Cognizable IPC Crimes using a Random Forest model.")
...
```

### 3. `utils/data_processing.py`

Handles data processing and feature engineering.

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(file):
    return pd.read_csv(file)

def preprocess_data(data):
    le_state = LabelEncoder()
    le_district = LabelEncoder()

    data['State_Encoded'] = le_state.fit_transform(data['States/UTs'])
    data['District_Encoded'] = le_district.fit_transform(data['District'])
    
    data['Total_Violent_Crimes'] = data['Murder'] + data['Rape'] + data['Robbery']
    
    return data, le_state, le_district
...
```

## ⚠️ Troubleshooting & Error Handling

### 1. **Dependency Installation Errors (Python 3.13)**:
Some dependencies may not work well with Python 3.13.

**Solution**:
- **Downgrade Python**: Use Python 3.11 for compatibility:
  - Download Python 3.11 from [python.org](https://www.python.org/downloads/).
  - Create a new virtual environment using Python 3.11 and reinstall the dependencies.

### 2. **Missing CSV File or Incorrect Format**:
If the uploaded file is missing or in an incorrect format, you might encounter errors during data loading.

**Solution**:
- Ensure the file is named `crime_data.csv` and is located inside the `data/` folder.
- Check that the CSV contains the required columns: `States/UTs`, `District`, `Year`, `Murder`, `Rape`, `Total Cognizable IPC crimes`.

### 3. **Streamlit Not Running Properly**:
If `streamlit run app.py` doesn't work or shows errors:

**Solution**:
- **Check for Errors**: Ensure no error messages appear in the terminal. Common errors include missing libraries or incorrect paths.
- **Reinstall Dependencies**: If necessary, reinstall the dependencies in the virtual environment:
    ```bash
    pip install -r requirements.txt
    ```

### 4. **Model Performance Issues**:
If the model’s accuracy seems off or predictions don’t align with expectations:

**Solution**:
- **Check Data Quality**: Ensure the dataset is clean and contains relevant features.
- **Evaluate Hyperparameters**: The Random Forest model may benefit from hyperparameter tuning. Try adjusting the number of trees or other model parameters for better performance.

### 5. **Input Validation Errors**:
If users cannot enter valid values or the input form is not working:

**Solution**:
- Ensure the correct types of inputs are provided (e.g., numeric values for crime features, and category selections for State and District).
  
### 6. **General App Crashes or Freezes**:
If the Streamlit app freezes or crashes unexpectedly:

**Solution**:
- **Clear Cache**: Clear Streamlit’s cache with:
    ```bash
    streamlit cache clear
    ```
- **Restart App**: Restart the app by running:
    ```bash
    streamlit run app.py
    ```

## 📈 Next Steps

- **Deployment**: Consider deploying the app on **Streamlit Sharing**, **Heroku**, or **AWS** for public access.
- **Improvement**: Enhance the app by incorporating other models (e.g., XGBoost) or adding new features like time-series forecasting.
  
## 👥 Contributing

Feel free to fork this repository, raise issues, and submit pull requests. Contributions are welcome!

---

Thank you for using the **Crime Hotspot Prediction** app! 🚀
