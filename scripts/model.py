from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

def train_model(X_train, y_train):
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    # The evaluation metric: score = max(0, 100 - sqrt(MSE))
    score = max(0, 100 - np.sqrt(mse))
    print("Validation Score:", score)
    return score
