FROM python:3.11-slim

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgomp1 \
    libgl1 \         
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем Python библиотеки
RUN pip install --no-cache-dir \
    pandas==2.1.4 \
    scikit-learn==1.3.2 \
    sqlalchemy==2.0.23 \
    psycopg2-binary==2.9.9 \
    mlflow==2.11.3 \
    catboost==1.2.5 \
    matplotlib==3.8.0 \
    seaborn==0.13.0 \
    numpy==1.24.3

# Создаем рабочую директорию
WORKDIR /app

# Копируем необходимые файлы
COPY train_model.py model_trainer_mlflow.py ./

# Устанавливаем переменные окружения
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg

# Проверяем импорт
RUN python -c "from model_trainer_mlflow import ModelTrainerMLflow; print('ModelTrainerMLflow импортирован успешно')"

# Команда запуска
CMD ["python", "train_model.py"]