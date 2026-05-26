import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

print("⏳ Старт Етапу 5: NLP-аналіз...")
df = pd.read_csv('data/processed.csv')

# Текстове поле
df['text_metadata'] = df['track_name'].fillna('') + " " + df['artists'].fillna('') + " " + df['album_name'].fillna('')

# Функція очищення тексту
def clean_text(text):
    text = text.lower() # Нижній регістр
    text = re.sub(r'\(.*?\)|\[.*?\]', '', text) # Видаляємо дужки (Remastered, Explicit)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text) # Видаляємо спецсимволи
    return text.strip()

df['text_metadata'] = df['text_metadata'].apply(clean_text)

# Векторизація за ТЗ
tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), min_df=2)
tfidf_matrix = tfidf.fit_transform(df['text_metadata'])

# Зменшення розмірності через TruncatedSVD до dense-ембедінгів
svd = TruncatedSVD(n_components=10) # 10 для демо (у ТЗ просять 50, підлаштуємо під розмір mock-даних)
text_embeddings = svd.fit_transform(tfidf_matrix)

# Зберігаємо щільні текстові вектори
np.save("models/text_embeddings.npy", text_embeddings)
print("✅ NLP-аналіз завершено. Текстові ембедінги збережено у 'models/text_embeddings.npy'.")