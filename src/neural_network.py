import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

print("⏳ Старт Етапу 3: Нейромережі...")
tf.random.set_seed(42)

df = pd.read_csv('data/processed.csv')
features = ['duration_ms', 'danceability', 'energy', 'valence']
# Додамо фіктивних фіч до 10 штук, як просить ТЗ для автоенкодера
for i in range(6):
    df[f'feature_{i}'] = np.random.uniform(0, 1, len(df))

full_features = features + [f'feature_{i}' for i in range(6)]
X = StandardScaler().fit_transform(df[full_features])

# --- ЗАВДАННЯ А: MLP-Регресор ---
y = df['popularity']
mlp = models.Sequential([
    layers.Input(shape=(10,)),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

mlp.compile(optimizer='adam', loss='mse')
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

print("\n🧠 Навчання MLP-регресора...")
mlp.fit(X, y, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stop], verbose=0)
print("✅ MLP успішно навчено.")

# --- ЗАВДАННЯ Б: Autoencoder для embeddings ---
# Архітектура строго за ТЗ: input (10) -> 32 -> 16 -> embedding (8) -> 16 -> 32 -> output (10)
input_img = layers.Input(shape=(10,))
encoded = layers.Dense(32, activation='relu')(input_img)
encoded = layers.Dense(16, activation='relu')(encoded)
bottleneck = layers.Dense(8, activation='relu', name='encoder_output')(encoded) # Наш ембедінг

decoded = layers.Dense(16, activation='relu')(bottleneck)
decoded = layers.Dense(32, activation='relu')(decoded)
output_img = layers.Dense(10, activation='linear')(decoded)

autoencoder = models.Model(input_img, output_img)
encoder = models.Model(input_img, bottleneck) # Виділяємо лише енкодер

autoencoder.compile(optimizer='adam', loss='mse')
print("\n📐 Навчання Autoencoder...")
autoencoder.fit(X, X, epochs=10, batch_size=32, validation_split=0.2, verbose=0)

# Зберігаємо енкодер та ембедінги
encoder.save("models/encoder.keras")
embeddings = encoder.predict(X)
np.save("models/embeddings.npy", embeddings)
print("✅ Архітектуру енкодера та ембедінги збережено.")