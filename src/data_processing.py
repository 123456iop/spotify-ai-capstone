import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

# Створюємо папки проєкту, якщо їх немає
os.makedirs('data', exist_ok=True)
os.makedirs('models/supervised', exist_ok=True)

# --- Симуляція завантаження сирих даних, якщо файлу немає ---
if not os.path.exists('data/dataset.csv'):
    print("📝 Вихідний файл не знайдено. Генеруємо тестовий data/dataset.csv...")
    np.random.seed(42)
    mock_raw = pd.DataFrame({
        'track_id': [f'id_{i}' for i in range(1000)],
        'track_name': [f'Track {i} (Remastered)' if i % 10 == 0 else f'Track {i}' for i in range(1000)],
        'artists': [f'Artist {i%20}' for i in range(1000)],
        'album_name': [f'Album {i%50}' for i in range(1000)],
        'popularity': np.random.choice([0, 10, 50, 80], size=1000, p=[0.1, 0.2, 0.5, 0.2]),
        'duration_ms': np.random.normal(loc=200000, scale=50000, size=1000).astype(int),
        'danceability': np.random.uniform(0, 1, 1000),
        'energy': np.random.uniform(0, 1, 1000),
        'valence': np.random.uniform(0, 1, 1000),
        'track_genre': np.random.choice(['pop', 'rock', 'hip-hop', 'jazz'], size=1000)
    })
    # Додамо аномалії для перевірки очищення
    mock_raw.loc[0, 'duration_ms'] = 10000   # Занадто короткий (10 сек)
    mock_raw.loc[1, 'duration_ms'] = 1000000 # Занадто довгий (16 хв)
    mock_raw.to_csv('data/dataset.csv', index=False)

# 1. Завантаження даних
print("⏳ Завантаження датасету...")
df = pd.read_csv('data/dataset.csv')

# 2. Очищення даних
print(f"📊 Початкова кількість треків: {len(df)}")
df = df.dropna() # Видалення пропусків
df = df.drop_duplicates(subset=['track_id']) # Видалення дублікатів

# Фільтрація за тривалістю (від 30 сек до 15 хвилин)
df = df[(df['duration_ms'] >= 30000) & (df['duration_ms'] <= 900000)]
print(f"🧹 Кількість треків після базового очищення: {len(df)}")

# 3. Пошук прихованих аномалій за допомогою IsolationForest
audio_features = ['danceability', 'energy', 'valence']
iso_forest = IsolationForest(contamination=0.02, random_state=42)
anomalies = iso_forest.fit_predict(df[audio_features])
df = df[anomalies == 1]
print(f"🚀 Кількість треків після видалення аномалій: {len(df)}")

# 4. Описова статистика
print("\n📋 Статистика по аудіо-фічах:")
print(df[audio_features].describe())

# 5. Генерація графіків (EDA)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='valence', y='energy', hue='track_genre', alpha=0.6)
plt.title("Карта настроїв: Valence vs Energy")
plt.savefig('data/mood_map.png')
plt.close()

# 6. Збереження очищеного датасету
df.to_csv('data/processed.csv', index=False)
print("✅ Очищений датасет збережено у 'data/processed.csv'")