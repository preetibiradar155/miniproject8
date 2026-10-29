import pickle
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# MLflow Experiment
mlflow.set_experiment("House-Price-Comparison")

# 1. Data - Load ya Create
# Agar tumhare pass house.csv hai toh ye use karo, nahi toh sample data banega
try:
    df = pd.read_csv("house.csv")
    X = df.drop("Price", axis=1)
    y = df["Price"]
except:
    print("Creating sample data...")
    np.random.seed(42)
    n = 1000
    X = pd.DataFrame({
        "Area": np.random.randint(500, 4000, n),
        "Bedrooms": np.random.randint(1, 6, n),
        "Age": np.random.randint(0, 30, n),
        "Location": np.random.randint(1, 10, n)
    })
    y = X["Area"]*0.2 + X["Bedrooms"]*100 + X["Location"]*50 - X["Age"]*5 + np.random.normal(0, 50, n)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear_Regression": LinearRegression(),
    "Random_Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient_Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = []

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print(f"\nModel: {name}")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R2 Score: {r2:.4f}")

        mlflow.log_param("model", name)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2_Score", r2)
        mlflow.sklearn.log_model(model, name)

        results.append([name, mae, rmse, r2])

# --- MODEL COMPARISON ---
print("\n--- MODEL COMPARISON ---")
comparison_df = pd.DataFrame(results, columns=["Model", "MAE", "RMSE", "R2_Score"])
comparison_df = comparison_df.sort_values(by="R2_Score", ascending=False)
print(comparison_df.to_string(index=False))

# Best Model
best_row = comparison_df.iloc[0]
best_model_name = best_row["Model"]
best_r2 = best_row["R2_Score"]

print(f"\nBest Model: {best_model_name}")
print(f"Best Model R2 Score: {best_r2:.4f}")

# Save best model
best_model_obj = models[best_model_name]
with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model_obj, f)
print("Saved: best_model.pkl")

comparison_df.to_csv("model_comparison.csv", index=False)
print("Saved: model_comparison.csv")
print("MLflow tracking completed successfully!")