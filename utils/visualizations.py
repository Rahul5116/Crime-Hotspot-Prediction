import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_crime_distribution(data, target='Total Cognizable IPC crimes'):
    """Plot the distribution of the target variable."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data[target], bins=30, kde=True, ax=ax)
    ax.set_xlabel('Total Crimes')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Total Cognizable IPC Crimes')
    return fig

def plot_correlation_matrix(data):
    """Plot the correlation matrix for numerical features."""
    fig, ax = plt.subplots(figsize=(10, 6))
    corr = data.select_dtypes(include=['int64', 'float64']).corr()
    sns.heatmap(corr, cmap='coolwarm', annot=False, ax=ax)
    ax.set_title('Correlation Matrix')
    return fig

def plot_feature_importance(feature_importance, feature_names):
    """Plot the top 10 feature importances."""
    fig, ax = plt.subplots(figsize=(8, 5))
    feature_importance_series = pd.Series(feature_importance, index=feature_names).sort_values(ascending=False)[:10]
    sns.barplot(x=feature_importance_series, y=feature_importance_series.index, ax=ax)
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    ax.set_title('Top 10 Feature Importance')
    return fig