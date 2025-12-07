import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')
from .aggregators import (
        BureauAggregator, PreviousApplicationAggregator, 
        InstallmentsAggregator, CreditCardAggregator, PosCashAggregator)
from .datapolish_log import DataPolish
from .datapolish_log import Hypothesis8, Hypothesis9, Hypothesis10

class HomeCreditFeatureEngineer:
    """
    Главный класс для полного пайплайна обработки данных
    
    Этапы:
    1. Загрузка данных
    2. Очистка и предобработка
    3. Агрегация вспомогательных таблиц
    4. Мердж всех фич
    5. Применение DataPolish
    6. Генерация гипотез 8-10
    7. Сохранение результата
    """
    
    def __init__(self, verbose: bool = True, sample_size: Optional[int] = None):
        self.verbose = verbose
        self.sample_size = sample_size
        self.dataframes = {}
        self.aggregated_features = {}
        
        # Инициализация агрегаторов
        self.bureau_agg = BureauAggregator(verbose=verbose)
        self.previous_app_agg = PreviousApplicationAggregator(verbose=verbose)
        self.installments_agg = InstallmentsAggregator(verbose=verbose)
        self.credit_card_agg = CreditCardAggregator(verbose=verbose)
        self.pos_cash_agg = PosCashAggregator(verbose=verbose)
        
        # Инициализация трансформеров
        self.datapolish = None
        self.hypothesis8 = Hypothesis8(verbose=verbose)
        self.hypothesis9 = Hypothesis9(verbose=verbose)
        self.hypothesis10 = Hypothesis10(verbose=verbose)
    
    def log(self, message: str):
        if self.verbose:
            print(f"[FeatureEngineer] {message}")
    
    def load_data(self, data_dict: Dict[str, pd.DataFrame]):
        """Загрузка данных из словаря DataFrame"""
        self.log("Загрузка данных")
        
        for name, df in data_dict.items():
            if df is not None and not df.empty:
                df.columns = df.columns.str.lower()
                
                if self.sample_size and len(df) > self.sample_size:
                    df = df.sample(self.sample_size, random_state=42)
                    self.log(f"  Взята выборка: {self.sample_size} строк")
                
                self.dataframes[name] = df
                self.log(f" {name}: {df.shape[0]} строк, {df.shape[1]} колонок")
            else:
                self.log(f" {name}: данные отсутствуют или пустые")
                self.dataframes[name] = pd.DataFrame()
        
        if 'application_train' not in self.dataframes or self.dataframes['application_train'].empty:
            raise ValueError("Обязательная таблица 'application_train' не найдена или пустая!")
        
        return self
    
    def preprocess_application(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """Предобработка основной таблицы (train или test)"""
        self.log(f"Предобработка данных (is_train={is_train})")
        
        df = df.copy()
        
        # 1. Заполнение пропусков для общих колонок
        fill_operations = [
            ('amt_goods_price', 'median'),
            ('name_type_suite', 'Unknown'),
            ('cnt_fam_members', 'median'),
            ('days_last_phone_change', 'median'),
            ('amt_annuity', 'median'),
        ]
        
        for col, method in fill_operations:
            if col in df.columns:
                if method == 'median':
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(method)
                self.log(f"    Заполнено {col}")
        
        social_cols = ['obs_30_cnt_social_circle', 'def_30_cnt_social_circle', 
                      'obs_60_cnt_social_circle', 'def_60_cnt_social_circle']
        
        for col in social_cols:
            if col in df.columns:
                df[f'{col}_missing'] = df[col].isnull().astype(int)
                df[col] = df[col].fillna(0)
        
        bureau_cols = ['amt_req_credit_bureau_hour', 'amt_req_credit_bureau_day', 
                      'amt_req_credit_bureau_week', 'amt_req_credit_bureau_mon', 
                      'amt_req_credit_bureau_qrt', 'amt_req_credit_bureau_year']
        
        for col in bureau_cols:
            if col in df.columns:
                df[f'{col}_missing'] = df[col].isnull().astype(int)
                df[col] = df[col].fillna(0)
        
        ext_cols = ['ext_source_1', 'ext_source_2', 'ext_source_3']
        for col in ext_cols:
            if col in df.columns:
                if col in ['ext_source_1', 'ext_source_3']:
                    df[f'{col}_missing'] = df[col].isnull().astype(int)
                    df[col] = df[col].fillna(df[col].median())
                elif col == 'ext_source_2':
                    df[col] = df[col].fillna(df[col].median())
        
        if 'own_car_age' in df.columns:
            df['own_car_age_missing'] = df['own_car_age'].isnull().astype(int)
            df['own_car_age'] = df['own_car_age'].fillna(0)
        
        if 'occupation_type' in df.columns:
            df['occupation_type_missing'] = df['occupation_type'].isnull().astype(int)
            df['occupation_type'] = df['occupation_type'].fillna('Unknown')
        
        self.log(f" Обработано {df.shape[1]} колонок")
        return df
    
    def run_aggregations(self):
        """Запуск всех агрегаторов"""
        self.log("Запуск агрегаций")
        
        # 1. Bureau агрегация
        if 'bureau' in self.dataframes and 'bureau_balance' in self.dataframes:
            if not self.dataframes['bureau'].empty and not self.dataframes['bureau_balance'].empty:
                bureau_features = self.bureau_agg.process(
                    self.dataframes['bureau'], 
                    self.dataframes['bureau_balance']
                )
                self.aggregated_features['bureau'] = bureau_features
            else:
                self.log(" Bureau: данные отсутствуют или пустые")
        
        # 2. Previous applications агрегация
        if 'previous_application' in self.dataframes:
            if not self.dataframes['previous_application'].empty:
                previous_app_features = self.previous_app_agg.process(
                    self.dataframes['previous_application']
                )
                self.aggregated_features['previous_application'] = previous_app_features
            else:
                self.log(" Previous applications: данные отсутствуют или пустые")
        
        # 3. Installments агрегация
        if 'installments_payments' in self.dataframes:
            if not self.dataframes['installments_payments'].empty:
                installments_features = self.installments_agg.process(
                    self.dataframes['installments_payments']
                )
                self.aggregated_features['installments'] = installments_features
            else:
                self.log(" Installments: данные отсутствуют или пустые")
        
        if 'credit_card_balance' in self.dataframes:
            if not self.dataframes['credit_card_balance'].empty:
                credit_card_features = self.credit_card_agg.process(
                    self.dataframes['credit_card_balance']
                )
                self.aggregated_features['credit_card'] = credit_card_features
            else:
                self.log(" Credit card: данные отсутствуют или пустые")
        
        if 'pos_cash_balance' in self.dataframes:
            if not self.dataframes['pos_cash_balance'].empty:
                pos_cash_features = self.pos_cash_agg.process(
                    self.dataframes['pos_cash_balance']
                )
                self.aggregated_features['pos_cash'] = pos_cash_features
            else:
                self.log(" POS cash: данные отсутствуют или пустые")
        
        self.log(f"Сгенерировано {len(self.aggregated_features)} наборов фич")
        return self
    
    def merge_all_features(self, main_df: pd.DataFrame) -> pd.DataFrame:
        """Мердж всех агрегированных фич с основной таблицей"""
        self.log("Объединение всех фич")
        
        final_df = main_df.copy()
        initial_cols = final_df.shape[1]
        
        for name, features_df in self.aggregated_features.items():
            if not features_df.empty and 'sk_id_curr' in features_df.columns and 'sk_id_curr' in final_df.columns:
                before = final_df.shape[1]
                final_df = final_df.merge(features_df, on='sk_id_curr', how='left')
                added = final_df.shape[1] - before
                self.log(f" + {name}: добавлено {added} фич")
            else:
                self.log(f"  - {name}: пропущено (отсутствуют ключевые колонки)")
        
        final_df = final_df.fillna(0)
        
        total_added = final_df.shape[1] - initial_cols
        self.log(f"Всего добавлено фич: {total_added}")
        self.log(f"Итоговый размер: {final_df.shape}")
        
        return final_df
    
    def apply_datapolish(self, df: pd.DataFrame) -> pd.DataFrame:
        """Применение DataPolish трансформера"""
        self.log("Применение DataPolish...")
        
        self.datapolish = DataPolish(
            max_na_ratio=0.7,
            low_variance_quantile=0.01,
            corr_threshold=0.95,
            handle_outliers=True,
            outlier_method='clip',
            log_transform_skew=2.0,
            scale_numeric=False,
            verbose=self.verbose
        )
        
        target = None
        if 'target' in df.columns:
            target = df['target'].copy()
        
        df_cleaned = self.datapolish.fit_transform(df)
        
        if target is not None:
            df_cleaned['target'] = target.values
        
        self.log(f" После DataPolish: {df_cleaned.shape}")
        return df_cleaned
    
    def apply_hypotheses(self, df: pd.DataFrame) -> pd.DataFrame:
        """Применение гипотез 8-10"""
        self.log("Применение гипотез 8-10...")
        
        df_with_hypotheses = df.copy()
        
        df_with_hypotheses = self.hypothesis8.transform(df_with_hypotheses)
        self.log(f"  + Гипотеза 8: добавлено {len(self.hypothesis8.features_added)} фич")
        
        df_with_hypotheses = self.hypothesis9.transform(df_with_hypotheses)
        self.log(f"  + Гипотеза 9: добавлено {len(self.hypothesis9.features_added)} фич")
        
        df_with_hypotheses = self.hypothesis10.transform(df_with_hypotheses)
        self.log(f"  + Гипотеза 10: добавлено {len(self.hypothesis10.features_added)} фич")
        
        self.log(f" После гипотез: {df_with_hypotheses.shape}")
        return df_with_hypotheses
    
    def run_full_pipeline(self) -> pd.DataFrame:
        """Запуск полного пайплайна обработки"""
        self.log("=" * 60)
        self.log("ЗАПУСК ПОЛНОГО ПАЙПЛАЙНА")
        self.log("=" * 60)
        
        # 1. Предобработка основной таблицы
        train_processed = self.preprocess_application(
            self.dataframes['application_train'], 
            is_train=True
        )
        
        # 2. Запуск агрегаций
        self.run_aggregations()
        
        # 3. Мердж всех фич
        train_final = self.merge_all_features(train_processed)
        
        # 4. DataPolish
        train_cleaned = self.apply_datapolish(train_final)
        
        # 5. Гипотезы
        train_with_hypotheses = self.apply_hypotheses(train_cleaned)
        
        self.log("=" * 60)
        self.log("ПАЙПЛАЙН УСПЕШНО ЗАВЕРШЁН!")
        self.log(f"Итоговый датасет: {train_with_hypotheses.shape}")
        self.log("=" * 60)
        
        return train_with_hypotheses
    
    def get_statistics(self) -> Dict:
        """Получение статистики по пайплайну"""
        stats = {
            'tables_loaded': len(self.dataframes),
            'aggregators_used': len(self.aggregated_features),
            'total_features_generated': sum(
                len(features.columns) for features in self.aggregated_features.values()
            ) if self.aggregated_features else 0,
        }
        
        return stats