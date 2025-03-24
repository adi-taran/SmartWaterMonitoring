import pandas as pd


def load_data():
    # Load training and testing datasets
    train_data = pd.read_csv('data/train.csv')
    test_data = pd.read_csv('data/test.csv')
    return train_data, test_data


def preprocess_data(train_data, test_data):
    # 1. Handle Missing Values in Numerical Columns
    numerical_cols_train = train_data.select_dtypes(include=['number']).columns
    numerical_cols_test = test_data.select_dtypes(include=['number']).columns

    # Exclude the target 'Water_Consumption' for train_data during imputation
    if 'Water_Consumption' in numerical_cols_train:
        numerical_cols_train = numerical_cols_train.drop('Water_Consumption')

    train_data[numerical_cols_train] = train_data[numerical_cols_train].fillna(
        train_data[numerical_cols_train].mean()
    )
    test_data[numerical_cols_test] = test_data[numerical_cols_test].fillna(
        test_data[numerical_cols_test].mean()
    )

    # 2. Encode Categorical Variables (All object columns except Timestamp)
    categorical_cols = train_data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col == 'Timestamp':
            continue

        # Try to convert column to numeric:
        converted_train = pd.to_numeric(train_data[col], errors='coerce')
        if converted_train.notnull().all():
            # If conversion is successful for every value, convert both datasets
            train_data[col] = pd.to_numeric(train_data[col])
            test_data[col] = pd.to_numeric(test_data[col], errors='coerce')
        else:
            # Otherwise, create a union mapping from train and test values
            all_values = pd.concat([train_data[col], test_data[col]], axis=0).unique()
            mapping = {val: idx for idx, val in enumerate(all_values)}
            train_data[col] = train_data[col].map(mapping)
            test_data[col] = test_data[col].map(mapping)

    # 3. Extract Features from Timestamp (e.g., Month)
    # Assuming the Timestamp format is 'day/month/year hour'; adjust the format if necessary.
    train_data['Month'] = pd.to_datetime(
        train_data['Timestamp'], format='%d/%m/%Y %H', errors='coerce'
    ).dt.month
    test_data['Month'] = pd.to_datetime(
        test_data['Timestamp'], format='%d/%m/%Y %H', errors='coerce'
    ).dt.month

    # Original Timestamp is preserved for submission purposes.
    return train_data, test_data
