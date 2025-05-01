import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def avg_percentage_error(y_true, y_pred):
    mask = y_true != 0
    return 100 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

def main():
    X_train = pd.read_csv("data/X_train.csv")
    X_test  = pd.read_csv("data/X_test.csv")
    y_train = pd.read_csv("data/y_train.csv").squeeze().values
    y_test  = pd.read_csv("data/y_test.csv").squeeze().values

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    print("Training Model...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae   = mean_absolute_error(y_test, y_pred)
    rmse  = mean_squared_error(y_test, y_pred, squared=False)
    mpe   = avg_percentage_error(y_test, y_pred)

    print("\nEvaluation on test set:")
    print(f"MAE : {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"Avg % Error: {mpe:.2f}%")

    joblib.dump(model, "models/rating_predictor.pkl")
    print("\nModel saved.")

if __name__ == "__main__":
    main()