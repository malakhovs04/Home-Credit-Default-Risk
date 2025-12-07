import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

def safe_divide(a, b):
    return np.divide(a, b, out=np.zeros_like(a), where=b != 0)

class Hypothesis8(BaseEstimator, TransformerMixin):
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.features_added = []
    
    def log(self, message: str):
        if self.verbose:
            print(f"[Hypothesis8] {message}")
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        self.log("Starting transformation")
        df = X.copy()
        self.features_added = []
        
        # 1. МЕТРИКИ НЕДОПЛАТ
        if all(col in df.columns for col in ['ip_amt_instalment_sum', 'ip_amt_payment_sum']):
            df['ip_underpayment_total'] = df['ip_amt_instalment_sum'] - df['ip_amt_payment_sum']
            df['ip_underpayment_ratio'] = safe_divide(df['ip_underpayment_total'], df['ip_amt_instalment_sum'])
            df['ip_paid_ratio'] = safe_divide(df['ip_amt_payment_sum'], df['ip_amt_instalment_sum'])
            self.features_added.extend(['ip_underpayment_total', 'ip_underpayment_ratio', 'ip_paid_ratio'])
            self.log("Added underpayment metrics")
        
        # 2. МЕТРИКИ ПРОСРОЧЕК ПО ДНЯМ
        days_cols = [col for col in df.columns if 'days_entry_payment' in col or 'days_instalment' in col]
        if days_cols:
            df['ip_max_delay'] = df[days_cols].max(axis=1, skipna=True) - df[days_cols].min(axis=1, skipna=True)
            df['ip_avg_delay'] = df[days_cols].mean(axis=1, skipna=True)
            self.features_added.extend(['ip_max_delay', 'ip_avg_delay'])
        
        # 3. ПРИЗНАКИ РЕСТРУКТУРИЗАЦИИ
        version_cols = [col for col in df.columns if col.startswith('ip_version_')]
        if len(version_cols) > 1:
            df['ip_restructured_flag'] = (df[version_cols].sum(axis=1) > 1).astype(int)
            df['ip_version_diversity'] = df[version_cols].gt(0).sum(axis=1)
            self.features_added.extend(['ip_restructured_flag', 'ip_version_diversity'])
        
        self.log(f"Added {len(self.features_added)} features")
        return df

class Hypothesis9(BaseEstimator, TransformerMixin):
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.features_added = []
        self.ext_weights = {'ext_source_1': 3, 'ext_source_2': 2, 'ext_source_3': 5}
    
    def log(self, message: str):
        if self.verbose:
            print(f"[Hypothesis9] {message}")
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        self.log("Starting transformation")
        df = X.copy()
        self.features_added = []
        
        ext_cols = ['ext_source_1', 'ext_source_2', 'ext_source_3']
        available_cols = [col for col in ext_cols if col in df.columns]
        
        if available_cols:
            # БАЗОВЫЕ СТАТИСТИКИ
            df['ext_mean'] = df[available_cols].mean(axis=1)
            df['ext_std'] = df[available_cols].std(axis=1)
            df['ext_max'] = df[available_cols].max(axis=1)
            df['ext_min'] = df[available_cols].min(axis=1)
            df['ext_range'] = df['ext_max'] - df['ext_min']
            
            # ВЗВЕШАННОЕ СРЕДНЕЕ
            weights_sum = sum(self.ext_weights[col] for col in available_cols)
            weighted_sum = sum(df[col].fillna(0) * self.ext_weights[col] for col in available_cols)
            df['ext_weight'] = weighted_sum / weights_sum
            
            df['ext_missing_count'] = df[ext_cols].isnull().sum(axis=1)
            
            self.features_added = ['ext_mean', 'ext_std', 'ext_max', 'ext_min', 'ext_range', 'ext_weight', 'ext_missing_count']
            self.log(f"Added {len(self.features_added)} external source features")
        else:
            self.log("No external source columns found")
        
        return df

class Hypothesis10(BaseEstimator, TransformerMixin):
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.features_added = []
    
    def log(self, message: str):
        if self.verbose:
            print(f"[Hypothesis10] {message}")
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        self.log("Starting transformation")
        df = X.copy()
        self.features_added = []
        
        # ФИНАНСОВЫЕ КОЭФФИЦИЕНТЫ
        if 'amt_credit' in df.columns and 'amt_income_total' in df.columns:
            df['credit_income_ratio'] = safe_divide(df['amt_credit'], df['amt_income_total'])
            df['annuity_income_ratio'] = safe_divide(df['amt_annuity'], df['amt_income_total'])
            self.features_added.extend(['credit_income_ratio', 'annuity_income_ratio'])
        
        if 'amt_credit' in df.columns and 'amt_annuity' in df.columns:
            df['credit_annuity_ratio'] = safe_divide(df['amt_credit'], df['amt_annuity'])
            self.features_added.append('credit_annuity_ratio')
        
        if all(col in df.columns for col in ['amt_goods_price', 'amt_credit']):
            df['goods_credit_diff'] = df['amt_goods_price'] - df['amt_credit']
            df['goods_credit_ratio'] = safe_divide(df['amt_goods_price'], df['amt_credit'])
            self.features_added.extend(['goods_credit_diff', 'goods_credit_ratio'])
        
        # СЕМЕЙНЫЕ МЕТРИКИ
        if all(col in df.columns for col in ['amt_income_total', 'cnt_fam_members']):
            df['income_per_person'] = safe_divide(df['amt_income_total'], df['cnt_fam_members'])   
            self.features_added.append('income_per_person')       
        
        if all(col in df.columns for col in ['cnt_children', 'cnt_fam_members']):
            df['children_ratio'] = safe_divide(df['cnt_children'], df['cnt_fam_members'])                                           
            self.features_added.append('children_ratio')
        
        # ВОЗРАСТ И СТАЖ 
        if 'days_birth' in df.columns:
            df['age_years'] = (-df['days_birth']) / 365.25
            self.features_added.append('age_years')
        
        if 'days_employed' in df.columns:
            df['days_employed_anomaly'] = (df['days_employed'] >= 365243).astype(int)
            df['employment_years'] = np.where(df['days_employed'] >= 365243, 0, (-df['days_employed']) / 365.25)
            self.features_added.extend(['employment_years', 'days_employed_anomaly'])
        
        # КОМБИНИРОВАННЫЕ МЕТРИКИ
        if all(col in df.columns for col in ['employment_years', 'age_years']):
            df['employment_age_ratio'] = safe_divide(df['employment_years'], df['age_years'])
            df['career_start_age'] = df['age_years'] - df['employment_years']
            self.features_added.extend(['employment_age_ratio', 'career_start_age'])
        
        self.log(f"Added {len(self.features_added)} demographic/financial features")
        return df