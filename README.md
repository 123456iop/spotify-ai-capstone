1. Встановіть усі потрібні бібліотеки:
pip install -r requirements.txt

2. Запустіть наступні команди:
 1.Очищення даних, EDA та генерація processed.csv
python src/data_processing.py
 2.Навчання моделей машинного навчання (прогноз популярності)
python src/supervised.py
 3.Навчання нейромереж (регресор та Autoencoder)
python src/neural_network.py
 4.Кластеризація треків (пошук музичних груп)
python src/clustering.py
 5.NLP-аналіз текстів (TF-IDF для назв треків/артистів)
python src/nlp_analysis.py

Після успішного завантаження, у папці models повинні 
з'явитися відповідні файли.

3. Запуск веб додатку
streamlit run app/main.py

Бажано, але необов'язково:
Встановити віртуальне середовище, щоб уникнути 
конфлікту бібліотек:
1.# Створення віртуального середовища
python -m venv .venv
2.# Активація (для Windows):
.venv\Scripts\activate або Активація (для macOS/Linux):
source .venv/bin/activate