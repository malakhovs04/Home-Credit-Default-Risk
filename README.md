# Прогнозирование риска дефолта по кредитам Home Credit

![Home Credit Default Risk](https://img.shields.io/badge/Kaggle-Competition-blue?style=flat-square) ![Python](https://img.shields.io/badge/Python-3.11-green?style=flat-square) ![Libraries](https://img.shields.io/badge/Libraries-CatBoost%20%7C%20LightGBM%20%7C%20XGBoost-yellow?style=flat-square) ![Metric](https://img.shields.io/badge/Metric-AUC--ROC-red?style=flat-square)

## Описание проекта
Этот проект решает задачу [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) на Kaggle. Цель — предсказать вероятность дефолта клиента по кредиту на основе исторических данных. 

**Ключевые цели:**
- Провести разведочный анализ данных (EDA).
- Разработать и протестировать 10 гипотез для улучшения модели.
- Построить базовую модель и сравнить её с улучшенными версиями.
- Оценить качество по метрике **AUC-ROC**.

Проект включает предобработку данных, feature engineering, моделирование и ансамблирование. Результаты оцениваются локально и на Kaggle (private/public score).

## Данные
Данные из соревнования Kaggle (7 CSV-файлов):
- **application_train/test.csv**: Основные данные о клиентах (122 признака, 307511 строк в train).
- **bureau.csv**: Прошлые кредиты в других организациях.
- **bureau_balance.csv**: Ежемесячные балансы по кредитам.
- **POS_CASH_balance.csv**: Балансы по POS-кредитам.
- **credit_card_balance.csv**: Балансы по кредитным картам.
- **previous_application.csv**: История заявок в Home Credit.
- **installments_payments.csv**: История платежей по рассрочкам.
- **HomeCredit_columns_description.csv**: Описание столбцов.

Данные обрабатываются с учетом пропусков, агрегаций (one-hot/mean encoding для категорий, статистики для числовых: min/max/mean/count/sum) и объединения таблиц.

## Требования
- **Python**: 3.11+
- **Библиотеки**: 
  ```
  pandas, numpy, matplotlib, seaborn, scikit-learn, catboost, lightgbm, xgboost
  ```
- **Данные**: Скачайте с [Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk/data) и разместите в папке `data/`.

## Структура проекта
- **check_tables.ipynb**: Разведочный анализ (EDA) всех таблиц — статистики, связи, визуализации.
- **aggregation.ipynb**: Предобработка — обработка пропусков, агрегации, объединение таблиц.
- **model.ipynb**: Тестирование моделей (Logistic Regression, Random Forest, LightGBM, CatBoost, XGBoost). Выбрана **CatBoost** как основная.
- **submission.ipynb**: Базовая модель и первый сабмит на Kaggle (`submission_baseline.csv`).
- **hypothesis_*.ipynb**: Проверка 10 гипотез (каждый файл — одна гипотеза с кодом, результатами и сабмитами).

## Классы и модули
По рекомендации, для повышения удобства и модульности кода, реализованы отдельные классы для предобработки, feature engineering и обучения моделей. Эти классы построены на базе scikit-learn (наследуют от BaseEstimator и TransformerMixin), что позволяет интегрировать их в пайплайны.

- **Data_polisher.py**: Класс `DataPolish` для предобработки данных.
  - Удаляет колонки с высоким процентом пропусков (>70%), константные и низковариативные признаки.
  - Удаляет высоко коррелирующие признаки (>0.95).
  - Обрабатывает пропуски (медиана для числовых, мода для категориальных).
  - Обрабатывает выбросы (по IQR), логарифмирует скошенные признаки (skew >2.0).
  - Опционально масштабирует числовые признаки (StandardScaler).

- **Hypotheses.py**: Классы для тестирования гипотез через feature engineering (гипотезы 8-10).
  - `Hypothesis8`: Поведение в рассрочках — метрики недоплат, просрочек по дням, признаки реструктуризации.
  - `Hypothesis9`: Внешние скоринги — комбинации external_source.
  - `Hypothesis10`: Финансовые и демографические метрики — коэффициенты (e.g., credit_income_ratio, employment_age_ratio), семейные метрики, возраст/стаж.

- **ModelTrainer.py**: Класс `ModelTrainer` для обучения и оценки моделей.
  - Поддерживает CatBoost, LightGBM, XGBoost с early stopping.
  - Вычисляет метрики: ROC-AUC, PR-AUC, accuracy, precision, recall, F1 (с оптимальным порогом).
  - Генерирует графики: ROC/PR кривые, feature importance.
  - Создает сабмиты для Kaggle.

Эти классы позволяют легко комбинировать предобработку и фичи в пайплайнах, упрощая эксперименты с гипотезами.

## Выбор модели
Тестировали 5 моделей на локальном датасете. Результаты:

| Модель              | AUC-ROC (локально) | Accuracy | Precision | Recall | F1-Score |
|---------------------|--------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.7684            | 0.9201  | 0.5855   | 0.0228 | 0.0440  |
| Random Forest       | 0.7159            | 0.9195  | 0.5263   | 0.0020 | 0.0040  |
| LightGBM            | 0.7773            | 0.9201  | 0.5556   | 0.0344 | 0.0647  |
| CatBoost            | 0.7846            | 0.9206  | 0.5714   | 0.0550 | 0.1003  |
| XGBoost             | 0.7689            | 0.1264  | 0.5263   | 0.0725 | 0.1264  |

**CatBoost** показал лучшие результаты и выбран для базовой модели.

## Гипотезы и результаты
Протестировано 10 гипотез. Каждая включает предобработку, обучение и сабмит на Kaggle. Результаты (AUC-ROC на Kaggle):

| Гипотеза | Описание | Private Score | Public Score | Сабмит-файл |
|----------|----------|---------------|--------------|-------------|
| Baseline | Базовая модель (CatBoost без улучшений) | 0.77855 | 0.77336 | submission_baseline.csv |
| 1 | Обработка выбросов (замена >99% на медиану) | 0.77585 | 0.77291 | submission_baseline_1.csv |
| 1.2 | Исключение ключевых столбцов из обработки выбросов | 0.77585 | 0.77291 | submission_baseline_1_2.csv |
| 1.3 | Порог выбросов на 95% вместо 99% | 0.77585 | 0.77291 | submission_baseline_1_3.csv |
| 2 | Удаление коррелирующих признаков | 0.77772 | 0.77326 | submission_baseline_2.csv |
| 3.1 | Новые признаки: credit_to_income, annuity_to_income | 0.77998 | 0.77515 | submission_baseline_3_1.csv |
| 3.2 | Дополнительные новые признаки | 0.77889 | 0.77544 | submission_baseline_3_2.csv |
| 3.3 | Объединение признаков из 3.1 и 3.2 | 0.77983 | 0.77430 | submission_baseline_3_3.csv |
| 4 | Соединение таблиц без one-hot encoding | 0.77191 | 0.77484 | submission_baseline_4_1.csv |
| 5.1 | Ансамбль: CatBoost + LightGBM | 0.77998 | 0.77539 | submission_baseline_5_1.csv |
| 5.2 | Ансамбль с новыми признаками из гипотезы 3 | 0.77998 | 0.77515 | submission_baseline_5_2.csv |
| 6 | Ручная балансировка классов | 0.77428 | 0.77337 | submission_baseline_6.csv |
| 7 | Стратифицированная 5-фолдовая кросс-валидация | 0.77950 | 0.77695 | submission_baseline_7_1.csv |
| 7.1 | Обучение на всех тренировочных данных | 0.78048 | 0.77625 | submission_baseline_7.csv |
|8| Метрики недоплат, просрочек по дням| 0,76440 | 0,7667 |submission_Hypothesis8_Installments.csv |
|9| Внешние скоринги | 0,76641 | 0.76591 | submission_Hypothesis9_Installments.csv|
|10| Финансовые и демографические метрики| 0.77043 | 0.77449 | submission_Hypothesis10_Installments.csv |
|11|Соединяем 8 + 9 + 10 гипотезы вместе| 0,77300 | 0,77449 |submission_comb_gup.csv |