import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def data_exploration(data):
    """
    Explore the dataset to understand its structure, features, and distribution of data.

    Parameters:
    data (DataFrame): Input dataframe containing historical sales data.

    Returns:
    None
    """
    print("Dataset shape:", data.shape)
    print("\nFirst few rows of the dataset:")
    print(data.head())

    # Check for missing values
    print("\nMissing values:")
    print(data.isnull().sum())

    # Check for outliers (you may use visualization or statistical methods for this)

def data_preprocessing(data):
    """
    Preprocess the dataset by handling missing values, outliers, and converting categorical variables into numerical representations.

    Parameters:
    data (DataFrame): Input dataframe containing historical sales data.

    Returns:
    data_encoded (DataFrame): Preprocessed dataframe with categorical variables encoded.
    """
    # Handle missing values (fill with mean, median, mode, or use other imputation methods)
    data.fillna(data.mean(), inplace=True)

    # Handle outliers (you may choose to remove outliers or transform them)

    # Convert categorical variables into numerical representations using One-Hot Encoding
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    encoder = OneHotEncoder(sparse=False, drop='first')  # 'drop' parameter for removing redundant columns
    encoded_cols = pd.DataFrame(encoder.fit_transform(data[categorical_cols]))
    encoded_cols.columns = encoder.get_feature_names(categorical_cols)

    # Concatenate encoded columns with original dataframe and drop original categorical columns
    data_encoded = pd.concat([data, encoded_cols], axis=1).drop(categorical_cols, axis=1)

    return data_encoded

def train_model(X_train, y_train):
    """
    Train the selected model (Random Forest Regressor) on the training data.

    Parameters:
    X_train (DataFrame): Features of the training set.
    y_train (Series): Target variable of the training set.

    Returns:
    model: Trained machine learning model.
    """
    # Choose the model (Random Forest Regressor)
    model = RandomForestRegressor()

    # Train the model
    model.fit(X_train, y_train)

    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the performance of the trained model using appropriate evaluation metrics.

    Parameters:
    model: Trained machine learning model.
    X_test (DataFrame): Features of the testing set.
    y_test (Series): Target variable of the testing set.

    Returns:
    mae (float): Mean Absolute Error of the model.
    rmse (float): Root Mean Squared Error of the model.
    """
    # Make predictions on the testing set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    return mae, rmse

# Load the dataset
data = pd.read_csv('supermarket_sales.csv')

# 1. Data Exploration
data_exploration(data)

# 2. Data Preprocessing
data_encoded = data_preprocessing(data)

# 3. Feature Engineering
# Additional feature engineering can be performed here if required.

# 4. Model Selection and Training
# Split the dataset into features (X) and target variable (y)
X = data_encoded.drop(['Date', 'Time', 'Invoice ID', 'Total'], axis=1)  # Assuming these columns are not used for prediction
y = data_encoded['Total']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = train_model(X_train, y_train)

# 5. Model Evaluation
mae, rmse = evaluate_model(model, X_test, y_test)

print("\nModel Evaluation:")
print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", rmse)
