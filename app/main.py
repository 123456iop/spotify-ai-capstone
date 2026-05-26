import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Подключаем папку src, чтобы Streamlit видел наши модели рекомендаций
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from recommender_models import SpotifyRecommender

st.set_page_config(page_title="Spotify AI Engine", layout="wide", page_icon="🎵")

st.title("🎵 Музыкальная AI-Система Spotify Capstone")
st.markdown("Добро пожаловать в панель управления вашей рекомендательной экосистемой!")

# Проверяем наличие обработанных данных
if not os.path.exists('data/processed.csv'):
    st.error("❌ Сначала запустите скрипты из папки `src/` для генерации данных и эмбеддингов!")
else:
    # Загружаем данные
    df = pd.read_csv('data/processed.csv')

    # Инициализируем рекомендатель
    rec_engine = SpotifyRecommender(df)

    # Имитируем фиты признаков для работы веб-интерфейса
    mock_audio = np.random.rand(len(df), 4)
    rec_engine.fit_content_based(mock_audio)

    # Загружаем эмбеддинги для гибридной модели, если они созданы
    if os.path.exists('models/embeddings.npy') and os.path.exists('models/text_embeddings.npy'):
        ae_emb = np.load('models/embeddings.npy')
        text_emb = np.load('models/text_embeddings.npy')
        rec_engine.fit_hybrid(ae_emb, text_emb)

    # Виджеты интерфейса
    st.sidebar.header("🎛️ Настройки поиска")
    selected_track_name = st.sidebar.selectbox("Выберите трек для рекомендаций:", df['track_name'].tolist())

    rec_type = st.sidebar.radio(
        "Выберите алгоритм системы:",
        ["Content-Based (Аудио)", "Cluster-Based (Группы + Discovery)", "Hybrid (Нейросеть + Текст)"]
    )

    # Находим индекс выбранного трека
    track_idx = df[df['track_name'] == selected_track_name].index[0]
    current_track = df.iloc[track_idx]

    # Выводим инфо о текущей песне
    col1, col2, col3 = st.columns(3)
    col1.metric("🎵 Текущий трек", current_track['track_name'])
    col2.metric("👤 Исполнитель", current_track['artists'])
    if 'cluster_name' in current_track:
        col3.metric("🏷️ Музыкальный Стиль", current_track['cluster_name'])

    st.subheader("🚀 Результаты работы ИИ-движка:")

    # Вызов нужной модели
    if rec_type == "Content-Based (Аудио)":
        results = rec_engine.recommend_content_based(track_idx, n_recommendations=5)
    elif rec_type == "Cluster-Based (Группы + Discovery)":
        results = rec_engine.recommend_cluster_based(track_idx, n_recommendations=5)
    else:
        if hasattr(rec_engine, 'hybrid_embeddings') and rec_engine.hybrid_embeddings is not None:
            results = rec_engine.recommend_hybrid(track_idx, n_recommendations=5)
        else:
            st.warning("⚠️ Гибридная модель не инициализирована. Показаны базовые результаты.")
            results = rec_engine.recommend_content_based(track_idx, n_recommendations=5)

    # Красивая таблица результатов
    st.dataframe(results[['track_name', 'artists', 'popularity']], use_container_width=True)