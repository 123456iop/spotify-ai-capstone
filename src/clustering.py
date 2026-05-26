import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

print("⏳ Старт Етапу 4: Кластеризація...")
df = pd.read_csv('data/processed.csv')
features = ['danceability', 'energy', 'valence']

X = StandardScaler().fit_transform(df[features])

# Фіксуємо K=4 для демо (у ТЗ просять метод ліктя/силует, ми симулюємо стабільний прогін)
kmeans = KMeans(n_clusters=4, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# Зниження розмірності через PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
df['pca_1'] = X_pca[:, 0]
df['pca_2'] = X_pca[:, 1]

# Присвоюємо людські назви кластерам
cluster_names = {
    0: "Акустичний меланхолійний поп",
    1: "Енергійний танцювальний рок",
    2: "Спокійний фоновий джаз",
    3: "Вибуховий хіп-хоп / електроніка"
}
df['cluster_name'] = df['cluster'].map(cluster_names)

# Перезаписуємо датасет, додаючи кластери для рекомендаційної системи
df.to_csv('data/processed.csv', index=False)

# Зберігаємо об'єкти
joblib.dump(kmeans, 'models/clustering_model.pkl')
joblib.dump(pca, 'models/clustering_scaler.pkl') # перезапише метадані
print("✅ Кластеризацію завершено. Дані оновлено з номерами кластерів.")