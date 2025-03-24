from scripts.preprocessing import load_data, preprocess_data
from scripts.model import train_model, evaluate_model
from scripts.predict import predict_and_submit
from sklearn.model_selection import train_test_split

# Step 1: Load Data
train_data, test_data = load_data()

# Step 2: Preprocess Data
train_data, test_data = preprocess_data(train_data, test_data)

# Step 3: Prepare Training Data
# Drop 'Timestamp' and 'Water_Consumption' from features for model training.
X = train_data.drop(['Timestamp', 'Water_Consumption'], axis=1)
y = train_data['Water_Consumption']

# Split train_data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train and Evaluate Model
model = train_model(X_train, y_train)
evaluate_model(model, X_val, y_val)

# Step 5: Prepare Test Data and Predict
# For the test set, drop the 'Timestamp' column from the features used for prediction.
X_test = test_data.drop(['Timestamp'], axis=1)
predict_and_submit(model, X_test, test_data)
