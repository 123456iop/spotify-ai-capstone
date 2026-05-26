import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

print("⏳ Старт Етапу 2: Прогноз популярності...")
df = pd.read_csv('data/processed.csv')

# 1. Виділення ознак та цільової змінної
features = ['duration_ms', 'danceability', 'energy', 'valence']
X = df[features]
y = df['popularity']

# Розділення на train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабування ознак (тільки на train!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Навчання моделей
models = {
    "Baseline (Ridge)": Ridge(),
    "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=50, random_state=42)
}

print("\n📊 Оцінка моделей:")
best_r2 = -1
best_model = None

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    # Крос-валідація
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    print(f" - {name} | Середній CV R²: {cv_scores.mean():.4f}")

    if cv_scores.mean() > best_r2:
        best_r2 = cv_scores.mean()
        best_model = model

# 3. Підбір гіперпараметрів для найкращої моделі
print(f"\n⚙️ Оптимізація найкращої моделі...")
param_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
grid_search = GridSearchCV(best_model, param_grid, cv=3, scoring='r2')
grid_search.fit(X_train_scaled, y_train)

final_model = grid_search.best_estimator_
print(f"✅ Найкращі параметри: {grid_search.best_params_}")

# 4. Важливість ознак
if hasattr(final_model, 'feature_importances_'):
    importances = final_model.feature_importances_
    for f, imp in zip(features, importances):
        print(f"🔥 Важливість фічі {f}: {imp:.4f}")

# 5. Збереження
joblib.dump(final_model, 'models/supervised/best_model.pkl')
joblib.dump(scaler, 'models/supervised/scaler.pkl')
print("✅ Модель та скейлер збережено у 'models/supervised/'")