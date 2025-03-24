import pandas as pd

def predict_and_submit(model, X_test, test_data):
    # Predict the Water_Consumption for test data
    predictions = model.predict(X_test)

    # Create submission DataFrame using the preserved Timestamp column
    submission = pd.DataFrame({
        'Timestamp': test_data['Timestamp'],
        'Water_Consumption': predictions
    })
    submission.to_csv('submission.csv', index=False)
    print("Submission file 'submission.csv' created!")
