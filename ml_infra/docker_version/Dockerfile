# Dockerfile для home-credit-processor
FROM python:3.11-slim

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем Python библиотеки
RUN pip install --no-cache-dir \
    pandas==2.1.4 \
    scikit-learn==1.3.2 \
    psycopg2-binary==2.9.9 \
    sqlalchemy==2.0.23 \
    numpy==1.24.3

# Создаем рабочую директорию
WORKDIR /app

# Копируем код обработки
COPY ./process_data.py .

# Копируем модуль feature engineering (если есть)
COPY ./dags/home_credit_features /app/home_credit_features

# Устанавливаем переменные окружения
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["python", "process_data.py"]