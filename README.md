# Прогнозирование риска дефолта по кредитам Home Credit

## Описание проекта

Этот проект — решение задачи [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) на платформе Kaggle, целью которой является прогнозирование вероятности дефолта клиента по кредиту.
**Цели проекта:**
- Провести детальный разведочный анализ данных (EDA).
- Разработать и протестировать 10 гипотез для улучшения качества модели.
- Построить базовую модель и сравнить её с улучшенными версиями.
- Оценить модели по метрике AUC-ROC.

## Данные

Данные взяты из соревнования Kaggle и включают несколько CSV-файлов:
- **application_train.csv / application_test.csv**: Основные данные о клиентах (122 признака, 307511 строк в train).
- **bureau.csv**: Данные о предыдущих кредитах в других финансовых организациях.
- **bureau_balance.csv**: Ежемесячные балансы по кредитам из бюро.
- **POS_CASH_balance.csv**: Балансы по POS-кредитам и кредитам наличными.
- **credit_card_balance.csv**: Балансы по кредитным картам.
- **previous_application.csv**: История предыдущих заявок в Home Credit.
- **installments_payments.csv**: История платежей по рассрочкам.
- **HomeCredit_columns_description.csv**: Описание столбцов.

## Описание файлов
- **check_tables.ipynb** - данный файл посвящен разведочному анализу данных всех наших таблиц. Просмотрела основные моменты,  поняла логику и связь таблиц, можно переходить к работе дальше.
- **aggregation.ipynb** - работа с пропусками, агрегации данных (оne hot-mean encoding - для категориальных признаков, по числовым признакам находим основные статистики: min, max, mean, count, sum), обьединение таблиц.
- **model.ipynb** - тестируем и выбираем лучшую модель:
  
| Модель              | AUC-ROC (локально) | Accuracy             | Precision  | Recal | F-1 score|
|---------------------|--------------------|----------------------|------------|-------|----------|
| Logistic Regression | 0.7684             | 0.9201               |  0.5855    |0.0228 |0.0440    |
| Random Forest       | 0.7159             | 0.9195               |  0.5263    |0.0020 |0.0040    |
| LightGBM            | 0.7773             | 0.9201               |  0.5556    |0.0344 |0.0647    |
| CatBoost            | 0.7846             | 0.9206               |  0.5714    |0.0550 |0.1003    |
| XGBoost             | 0.7689             |0.1264                |  0.5263    |0.0725 |0.1264    |

для работы в дальнейшем я выбрала СatBoost 
- **summission.ipymb** -  содержит код для создания базовой модели и подготовки первого самбишена на Kagle. (**submission_baseline.csv**)
- **hypothesis_1.ipynb** - проверка гипотезы об обработке выбросов.
  
**submission_baseline_1.csv** - замена значений, превышающих 99-перцентиль, на медианну этого столбца

**submission_baseline_1_2.csv** - в перый раз я обработывала все колонки, теперь не включаю в обработку "sk_is_curr", "target", "sk_id_bureau", "sk_id_prev"

**submission_baseline_1_3.csv** - гипотеза о том, чтобы сделать порог в 95 перцентиль вместо 99

- **hypothesis_2.ipynb** - удаление коррелируемых признаков (**submission_baseline_2.csv**)
- **hypothesis_3.ipynb** - создание новых признаков
  
**submission_baseline_3_1.csv** - credit_to_income: Отношение суммы кредита к годовому доходу (amt_credit / amt_income_total)
annuity_to_income: Отношение аннуитетного платежа к годовому доходу (amt_annuity / amt_income_total) 

**submission_baseline_3_2.csv** - добавление еще признаков

**submission_baseline_3_3.csv** - обьединение результата двух предыдущих действий

- **hypothesis_4.ipynb** - Соединение таблиц без использования one-hot encoding (**submission_baseline_4_1.csv**)
- **hypothesis_5.ipynb** - Ансамблирование моделей

**submission_baseline_5_1.csv** - CatBoost + LightGBM

**submission_baseline_5_2.csv** - добавление новых признаков из 3 гипотезы
- **hypothesis_6.ipynb** - ручная балансировка классов (**submission_baseline_6.csv**)
- - **hypothesis_6.ipynb** - стратифицированная 5-ти фолдовая кросс-валидация (**submission_baseline_7_1.csv**)

**submission_baseline_7.csv** - обучение на всех тренировочных данных

  
| гипотеза                  | privat_score|public_score|
|---------------------------|-------------|------------|
|submission_baseline.csv    |0.77855      |0.77336     |
|submission_baseline_1.csv  |0.77585      |0.77291     |
|submission_baseline_1_2.csv|0.77585      |0.77291     |
|submission_baseline_1_3.csv|0.77585      |0.77291     |
|submission_baseline_2.csv  |0.777720     |0.77326     |
|submission_baseline_3_1.csv|0.77998      |0.77515     |
|submission_baseline_3_2.csv|0.77889      |0.77544     |
|submission_baseline_3_3.csv|0.77983      |0.77430     |
|submission_baseline_4_1.csv|0.77191      |0.77484     |
|submission_baseline_5_1.csv|0.77998      |0.77539     |
|submission_baseline_5_2.csv|0.77998      |0.77515     |
|submission_baseline_6.csv  |0.77428      |0.77337     |
|submission_baseline_7.csv  |0.78048      |0.77625     |
|submission_baseline_7_1.csv|0.77950      |0.77695     |

