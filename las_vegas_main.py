import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Function to load data
def load_data(filepath):
    return pd.read_csv(filepath,delimiter=';')

# Function to inspect the data
def inspect_data(df):
    print(df.info())
    print(df.describe())
    print(df.head())

# Univariate Analysis
def plot_histogram(df, column):
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True)
    plt.title(f'Histogram of {column}')
    plt.show()

def plot_boxplot(df, column):
    plt.figure(figsize=(10, 6))
    sns.boxplot(y=df[column])
    plt.title(f'Boxplot of {column}')
    plt.show()

# Bivariate Analysis
def plot_scatter(df, x_col, y_col):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df[x_col], y=df[y_col])
    plt.title(f'Scatter plot of {x_col} vs {y_col}')
    plt.show()

def plot_grouped_bar(df, x_col, y_col):
    plt.figure(figsize=(12, 8))
    sns.barplot(x=x_col, y=y_col, data=df, ci=None)
    plt.title(f'Average {y_col} by {x_col}')
    plt.xticks(rotation=45)
    plt.show()

# Multivariate Analysis
def plot_pairplot(df, columns):
    sns.pairplot(df[columns])
    plt.show()

# Function to encode categorical columns
def label_encode_columns(df, columns):
    le = LabelEncoder()
    for col in columns:
        if col in df.columns:  # Check if the column exists in the DataFrame
            df[col] = le.fit_transform(df[col].astype(str))
        else:
            print(f"Column not found: {col}")
    return df

# Function to plot the correlation matrix
def plot_correlation_matrix(df):
    # Optionally encode or select numeric columns here
    df = df.select_dtypes(include=[np.number])  # Ensure df is numeric if not pre-processed
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.show()


# Example usage
if __name__ == '__main__':
    df = load_data('LasVegasTripAdvisorReviews-Dataset.csv')
    inspect_data(df)
    plot_histogram(df, 'Score')
    plot_boxplot(df, 'Helpful votes')
    plot_scatter(df, 'Helpful votes', 'Score')
    plot_grouped_bar(df, 'Traveler type', 'Score')
    plot_pairplot(df, ['Score', 'Nr. reviews', 'Helpful votes'])
    categorical_columns = ['User country', 'Traveler type']
    df = label_encode_columns(df, categorical_columns)
    plot_correlation_matrix(df)
