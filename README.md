markdown
# Smart Water Monitoring Systems

This project aims to predict daily water consumption for individual households based on historical usage patterns, household characteristics, weather conditions, and conservation behaviors. It is developed as part of a HackerEarth Machine Learning challenge where the final submission file must contain two columns: `Timestamp` and `Water_Consumption`.

## Project Overview

Water scarcity is a pressing global issue. Traditional water meters only capture total usage and do not provide insights into consumption patterns. By leveraging machine learning, this project predicts household water consumption so that users can adopt effective conservation measures. The model is trained on historical data and uses a Random Forest regressor to make predictions.

## Project Structure

SmartWaterMonitoring/ │ ├── data/ │ ├── train.csv # Training data with historical usage, household info, etc. │ ├── test.csv # Test data without the target variable (Water_Consumption) │ └── sample_submission.csv # Sample submission file format │ ├── scripts/ │ ├── preprocessing.py # Data loading, preprocessing, feature engineering │ ├── model.py # Model training and evaluation functions │ ├── predict.py # Generating predictions and creating the submission file │ └── main.py # Main script to orchestrate the data pipeline │ ├── requirements.txt # Python dependencies for the project └── README.md # This file


## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/adi-taran/SmartWaterMonitoring.git
Open the Project in PyCharm: Open PyCharm and select the project folder SmartWaterMonitoring.

Set Up a Virtual Environment (Recommended): In PyCharm, create a new virtual environment or use an existing one.

Install Dependencies: Open the PyCharm terminal and run:

bash
pip install -r requirements.txt
How to Run the Project
Data Files Placement: Ensure that train.csv, test.csv, and sample_submission.csv are located in the data/ directory.

Run the Main Script: In PyCharm, open scripts/main.py and execute it (e.g., press Shift + F10 on Windows). The script will perform the following tasks:

Load and preprocess the data.

Split the training data for model training and validation.

Train a Random Forest model.

Evaluate the model using a custom metric.

Generate predictions for the test set.

Create a submission.csv file with the original Timestamp and predicted Water_Consumption.

Review the Submission: After execution, check that submission.csv is created in the project root following this format:

Timestamp,Water_Consumption
11/10/2014 16,306.1420999999999
12/10/2014 00,191.71720000000002
...
Dependencies
The project uses the following Python libraries:

pandas

numpy

scikit-learn

matplotlib

seaborn

These are all listed in the requirements.txt file.

Additional Information
Data Preprocessing: The preprocessing steps include handling missing values for numerical columns, encoding categorical features (while converting numeric-like strings where applicable), and extracting features (e.g., 'Month') from the Timestamp.

Model Training: A Random Forest Regressor from scikit-learn is used for training. The evaluation metric is defined as:

score = max(0, 100 - sqrt(MSE(actual, predicted)))
Submission File: The generated submission file retains the original Timestamp column along with the predicted water consumption.

Contributing
If you'd like to contribute to this project, please fork the repository and submit your pull requests. For any issues or questions, feel free to open an issue in the repository.

License
This project is licensed under the MIT License.

Happy coding and good luck with your submission!


---


