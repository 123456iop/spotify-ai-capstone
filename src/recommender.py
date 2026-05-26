import pandas as pd
import numpy as np
from recommender_models import SpotifyRecommender

# --- ЕМУЛЯЦІЯ ДАНИХ З ПОПЕРЕДНІХ ЕТАПІВ ---
np.random.seed(42)
n_tracks = 50

# Симулюємо базовий очищений датасет (Етап 1 та Етап 4 з кластерами)
mock_df = pd.DataFrame({
    'track_id': [f'track_{i}' for i in range(n_tracks)],
    'track_name': [f'Song Named {i}' for i in range(n_tracks)],
    'artist': [f'Artist {i % 5}' for i in range(n_tracks)],
    'cluster': np.random.randint(0, 4, n_tracks)  # Результат кластеризації K-Means
})

# Симулюємо ознаки:
mock_audio_features = np.random.rand(n_tracks, 10)  # 10 оригінальних аудіо-фіч
mock_ae_embeddings = np.random.rand(n_tracks, 8)    # 8 вимірів від Autoencoder
mock_text_embeddings = np.random.rand(n_tracks, 50)  # 50 вимірів після TruncatedSVD (NLP)
# -----------------------------------------------------------------

print("⚙️ Ініціалізація єдиного рекомендера...")
recommender = SpotifyRecommender(mock_df)

print("🧠 Тренуємо моделі...")
recommender.fit_content_based(mock_audio_features)
recommender.fit_hybrid(mock_ae_embeddings, mock_text_embeddings)

# Вибираємо випадковий трек для тесту (наприклад, з індексом 5)
test_track_idx = 5
current_track = mock_df.iloc[test_track_idx]

print("\n" + "=" * 60)
print(f"🎧 Поточний трек: '{current_track['track_name']}' [Виконавець: {current_track['artist']}]")
print("=" * 60)

# 1. Тест Content-Based
print("\n🔥 СИСТЕМА 1: Рекомендації за аудіо-характеристиками (Content-Based)")
rec_1 = recommender.recommend_content_based(test_track_idx, n_recommendations=3)
print(rec_1[['track_name', 'artist']])

# 2. Тест Cluster-Based
print("\n🔥 СИСТЕМА 2: Рекомендації на основі кластерів (з Discovery Mode)")
rec_2 = recommender.recommend_cluster_based(test_track_idx, cluster_column='cluster', n_recommendations=4)
print(rec_2[['track_name', 'artist', 'cluster']])

# 3. Тест Hybrid
print("\n🔥 СИСТЕМА 3: Гібридні рекомендації (Нейромережа + Текст)")
rec_3 = recommender.recommend_hybrid(test_track_idx, n_recommendations=3)
print(rec_3[['track_name', 'artist']])
print("\n" + "=" * 60)