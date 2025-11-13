# 🔋 Battery Lifecycle Analysis & Ionic Conductivity Prediction

Welcome to my **100 Days of ML for Materials Science** journey! This repository explores machine learning applications in battery materials, focusing on:

- 📈 Predicting ionic conductivity of solid electrolytes
- 🔄 Analyzing battery capacity over charge cycles
- 🧠 Comparing instance-based vs model-based learning

## 🚀 Project Highlights

- Simulated dataset of LLZO-based materials
- Random Forest & Linear Regression models
- Feature importance analysis
- Cross-validation and synthetic data augmentation

## 📂 Structure


## 🛠️ Tools Used

- Python, Pandas, NumPy, Matplotlib
- Scikit-learn (RandomForest, LinearRegression)
- GitHub for version control

## 📅 100 Days of ML Progress

| Day | Topic                                      | Status |
|-----|--------------------------------------------|--------|
| 1   | Project setup & repo creation              | ✅     |
| 10  | Ionic conductivity prediction (RF model)   | ✅     |
| 15  | Battery cycle prediction (coming soon)     | 🔜     |
Code: 
# ================================================================
# 🧠 ML for Materials Science — Battery Degradation Modeling
# ================================================================

# --- 1. Import libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# --- 2. Create base dataset ---
data = {
    "Material": ["LLZO", "LLTO", "LATP", "LAGP", "PEO+LLZO", "LLZO-Al", "LLZO-Ga", "PEO+LLTO", "PEO+LATP", "LLZO+PIL"],
    "Density": [5.1, 4.8, 4.5, 4.6, 3.2, 5.05, 5.0, 3.1, 3.3, 3.4],
    "Bandgap": [5.4, 3.2, 4.1, 4.3, 5.0, 5.5, 5.6, 3.5, 4.2, 4.8],
    "IonicRadius": [0.76, 0.90, 0.74, 0.72, 0.80, 0.75, 0.76, 0.88, 0.73, 0.79],
    "Conductivity": [1.2e-3, 8.5e-4, 3.4e-4, 2.7e-4, 5.6e-4, 1.4e-3, 1.5e-3, 9.1e-4, 4.1e-4, 6.0e-4]
}
df = pd.DataFrame(data)

# Add baseline temperature to original data
df["Temperature"] = 25.0

# --- 3. Augment dataset with temperature, noise, and non-linear effects ---
def augment_dataset(df, n=5, noise_level=0.05):
    augmented = []
    for _, row in df.iterrows():
        for _ in range(n):
            new_row = row.copy()
            new_row["Temperature"] = np.random.uniform(20, 80)
            temp_factor = 1 + 0.01 * (new_row["Temperature"] - 25)
            new_row["Density"] += np.random.uniform(-noise_level, noise_level)
            new_row["Bandgap"] += np.random.uniform(-noise_level, noise_level)
            new_row["IonicRadius"] += np.random.uniform(-0.02, 0.02)
            new_row["Conductivity"] *= temp_factor

            # Non-linear degradation
            if new_row["Bandgap"] < 4.5:
                new_row["Conductivity"] *= 0.7

            # Add Gaussian noise
            new_row["Conductivity"] += np.random.normal(0, noise_level * row["Conductivity"])
            augmented.append(new_row)
    return pd.DataFrame(augmented)

aug_df = augment_dataset(df, n=5)
full_df = pd.concat([df, aug_df], ignore_index=True)
full_df.dropna(inplace=True)

# --- 4. Prepare features and target ---
X = full_df[["Density", "Bandgap", "IonicRadius", "Temperature"]]
y = full_df["Conductivity"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 5. Train models ---
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
y_rf_pred = rf_model.predict(X_test)

knn_model = KNeighborsRegressor(n_neighbors=3)
knn_model.fit(X_train, y_train)
y_knn_pred = knn_model.predict(X_test)

# --- 6. Evaluate models ---
def evaluate_model(name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"{name} Performance:\nR²: {r2:.3f}, MSE: {mse:.2e}, MAE: {mae:.2e}\n")

evaluate_model("Random Forest", y_test, y_rf_pred)
evaluate_model("kNN", y_test, y_knn_pred)

# --- 7. Visualizations ---
plt.figure(figsize=(6,5))
plt.scatter(y_test, y_rf_pred, color='blue', label='Random Forest', alpha=0.6)
plt.scatter(y_test, y_knn_pred, color='green', label='kNN', alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("True Conductivity (S/cm)")
plt.ylabel("Predicted Conductivity (S/cm)")
plt.title("True vs Predicted Conductivity")
plt.legend()
plt.grid(True)
plt.show()

# Error comparison
rf_mse = mean_squared_error(y_test, y_rf_pred)
knn_mse = mean_squared_error(y_test, y_knn_pred)
rf_r2 = r2_score(y_test, y_rf_pred)
knn_r2 = r2_score(y_test, y_knn_pred)

plt.figure(figsize=(6,4))
plt.bar(["Random Forest", "kNN"], [rf_mse, knn_mse], color=['blue', 'green'])
plt.title("Model Comparison: MSE")
plt.ylabel("Mean Squared Error")
plt.show()

plt.figure(figsize=(6,4))
plt.bar(["Random Forest", "kNN"], [rf_r2, knn_r2], color=['blue', 'green'])
plt.title("Model Comparison: R² Score")
plt.ylabel("R²")
plt.show()

# --- 8. Feature importance ---
plt.figure(figsize=(6,4))
plt.bar(X.columns, rf_model.feature_importances_, color='teal')
plt.title("Feature Importance (Random Forest)")
plt.ylabel("Importance")
plt.show()

# --- 9. Cross-validation ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rf_cv = cross_val_score(rf_model, X, y, cv=kf, scoring='r2')
lr_model = make_pipeline(StandardScaler(), LinearRegression())
lr_cv = cross_val_score(lr_model, X, y, cv=kf, scoring='r2')

print(f"Random Forest CV R²: {np.mean(rf_cv):.3f} +/- {np.std(rf_cv):.3f}")
print(f"Linear Regression CV R²: {np.mean(lr_cv):.3f} +/- {np.std(lr_cv):.3f}")

# --- 10. Summary ---
print("""
Summary:
- Enhanced synthetic dataset with temperature, non-linear degradation, and sensor noise.
- Compared kNN (instance-based) vs Random Forest (model-based).
- Visualized predictions and error metrics.
- Feature importance shows Temperature and Bandgap are key drivers.
- Ready for visual storytelling and short video explanation.
""")


## 📬 Contact

Feel free to reach out or contribute!  
**Author**: Apar Niraula  
**Email**: niraulaapar016@gmail.com  
**LinkedIn**: [Your LinkedIn URL]

