import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


class SpotifyRecommender:
    def __init__(self, df):
        # Зберігаємо копію датасету та скидаємо індекси для точного мапінгу
        self.df = df.copy().reset_index(drop=True)
        self.nn_content = None
        self.nn_hybrid = None
        self.audio_features = None
        self.hybrid_embeddings = None

    # --- СИСТЕМА 1: Content-Based ---
    def fit_content_based(self, audio_features):
        """Навчання NearestNeighbors на оригінальних аудіо-характеристиках треків"""
        self.audio_features = np.array(audio_features)
        self.nn_content = NearestNeighbors(metric="cosine")
        self.nn_content.fit(self.audio_features)

    def recommend_content_based(self, track_idx, n_recommendations=5):
        """Пошук N найближчих сусідів з виключенням самого треку-запиту"""
        query_vector = self.audio_features[track_idx].reshape(1, -1)
        distances, indices = self.nn_content.kneighbors(query_vector, n_neighbors=n_recommendations + 1)

        # Індекси рекомендованих треків (пропускаємо перший [0], бо це сам запит)
        rec_indices = indices[0][1:]
        return self.df.iloc[rec_indices]

    # --- СИСТЕМА 2: Cluster-Based ---
    def recommend_cluster_based(self, track_idx, cluster_column='cluster', n_recommendations=5, discovery_mode=True):
        """Рекомендації на основі кластерів з Етапу 4 + режим Discovery (70/30)"""
        current_cluster = self.df.loc[track_idx, cluster_column]

        # Треки з того самого кластера (виключаючи поточний)
        same_cluster_tracks = self.df[self.df[cluster_column] == current_cluster].drop(track_idx, errors='ignore')

        if not discovery_mode:
            return same_cluster_tracks.sample(min(n_recommendations, len(same_cluster_tracks)))

        # Discovery mode: 70% з того ж кластера, 30% з інших кластерів для різноманітності
        n_same = int(n_recommendations * 0.7)
        n_diff = n_recommendations - n_same

        diff_cluster_tracks = self.df[self.df[cluster_column] != current_cluster]

        # Безпечно вибираємо випадкові треки (враховуючи, що їх може бути менше ніж ліміт)
        rec_same = same_cluster_tracks.sample(min(n_same, len(same_cluster_tracks)))
        rec_diff = diff_cluster_tracks.sample(min(n_diff, len(diff_cluster_tracks)))

        return pd.concat([rec_same, rec_diff])

    # --- СИСТЕМА 3: Hybrid (З нейромережею та NLP) ---
    def fit_hybrid(self, autoencoder_embeddings, tfidf_embeddings):
        """L2-нормалізація та зважена конкатенація ембедінгів (70% аудіо + 30% текст)"""
        # 1. Нормалізуємо обидва набори ембедінгів
        audio_norm = normalize(autoencoder_embeddings, norm='l2')
        text_norm = normalize(tfidf_embeddings, norm='l2')

        hybrid_space = np.hstack([audio_norm * 0.7, text_norm * 0.3])

        self.hybrid_embeddings = hybrid_space
        self.nn_hybrid = NearestNeighbors(metric="cosine")
        self.nn_hybrid.fit(hybrid_space)

    def recommend_hybrid(self, track_idx, n_recommendations=5):
        """Пошук найближчих сусідів у гібридному просторі знань"""
        query_vector = self.hybrid_embeddings[track_idx].reshape(1, -1)
        distances, indices = self.nn_hybrid.kneighbors(query_vector, n_neighbors=n_recommendations + 1)

        rec_indices = indices[0][1:]
        return self.df.iloc[rec_indices]