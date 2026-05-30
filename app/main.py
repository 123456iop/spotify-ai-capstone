import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Підключаємо папку src, щоб Streamlit бачив наші моделі рекомендацій
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from recommender_models import SpotifyRecommender

st.set_page_config(page_title="Spotify AI Engine", layout="wide", page_icon="🎵")

st.title("🎵 Музична AI-Система Spotify Capstone")
st.markdown("Вітаємо в панелі керування вашою рекомендаційною екосистемою!")

# Перевіряємо наявність оброблених даних
if not os.path.exists('data/processed.csv'):
    st.error("❌ Спочатку запустіть скрипти з папки `src/` для генерації даних та ембеддингів!")
else:
    # Завантажуємо дані
    df = pd.read_csv('data/processed.csv')

    # Ініціалізуємо рекомендаційну систему
    rec_engine = SpotifyRecommender(df)

    # Імітуємо фіти ознак для роботи веб-інтерфейсу (заміни на реальні при потребі)
    mock_audio = np.random.rand(len(df), 4)
    rec_engine.fit_content_based(mock_audio)

    # Завантажуємо ембеддинги для гібридної моделі, якщо вони створені
    if os.path.exists('models/embeddings.npy') and os.path.exists('models/text_embeddings.npy'):
        ae_emb = np.load('models/embeddings.npy')
        text_emb = np.load('models/text_embeddings.npy')
        rec_engine.fit_hybrid(ae_emb, text_emb)

    # Віджети інтерфейсу
    st.sidebar.header("🎛️ Налаштування пошуку")
    selected_track_name = st.sidebar.selectbox("Оберіть трек для рекомендацій:", df['track_name'].tolist())

    rec_type = st.sidebar.radio(
        "Оберіть алгоритм системи:",
        ["Content-Based (Аудіо)", "Cluster-Based (Групи + Discovery)", "Hybrid (Нейромережа + Текст)"]
    )

    # Знаходимо індекс обраного треку
    track_idx = df[df['track_name'] == selected_track_name].index[0]
    current_track = df.iloc[track_idx]

    # Виводимо інформацію про поточну пісню
    col1, col2, col3 = st.columns(3)
    col1.metric("🎵 Поточний трек", current_track['track_name'])
    col2.metric("👤 Виконавець", current_track['artists'])
    if 'cluster_name' in current_track:
        col3.metric("🏷️ Музичний Стиль", current_track['cluster_name'])

    st.subheader("🚀 Результати роботи AI-рушія:")

    # Виклик потрібної моделі
    if rec_type == "Content-Based (Аудіо)":
        results = rec_engine.recommend_content_based(track_idx, n_recommendations=5)
    elif rec_type == "Cluster-Based (Групи + Discovery)":
        results = rec_engine.recommend_cluster_based(track_idx, n_recommendations=5)
    else:
        if hasattr(rec_engine, 'hybrid_embeddings') and rec_engine.hybrid_embeddings is not None:
            results = rec_engine.recommend_hybrid(track_idx, n_recommendations=5)
        else:
            st.warning("⚠️ Гібридна модель не ініціалізована. Показано базові результати.")
            results = rec_engine.recommend_content_based(track_idx, n_recommendations=5)

    # Гарна таблиця результатів
    results_display = results[['track_name', 'artists', 'popularity']].copy()
    results_display.columns = ['Назва треку', 'Виконавець', 'Популярність']
    st.dataframe(results_display, use_container_width=True)