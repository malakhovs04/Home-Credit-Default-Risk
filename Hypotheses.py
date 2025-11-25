import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings('ignore')

def safe_divide(a, b):
    return np.divide(a, b, out=np.zeros_like(a), where=b!=0)

class Hypothesis8(BaseEstimator, TransformerMixin):
    """
    ГИПОТЕЗА 8: ПОВЕДЕНИЕ В РАССРОЧКАХ
    
    Основная идея: Клиенты, которые регулярно задерживают платежи по рассрочкам,
    с высокой вероятностью будут проблемными и для новых кредитов.
    """
    
    def __init__(self):
        self.features_added = []
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        self.features_added = []
        
        # 1. МЕТРИКИ НЕДОПЛАТ
        if all(col in df.columns for col in ['ip_amt_instalment_sum', 'ip_amt_payment_sum']):
            df['ip_underpayment_total'] = df['ip_amt_instalment_sum'] - df['ip_amt_payment_sum']
            df['ip_underpayment_ratio'] = safe_divide(
                df['ip_underpayment_total'], 
                df['ip_amt_instalment_sum']
            )
            df['ip_paid_ratio'] = safe_divide(
                df['ip_amt_payment_sum'], 
                df['ip_amt_instalment_sum']
            )
            self.features_added.extend([
                'ip_underpayment_total', 'ip_underpayment_ratio', 'ip_paid_ratio'
            ])
        
        # 2. МЕТРИКИ ПРОСРОЧЕК ПО ДНЯМ
        days_cols = [col for col in df.columns if 'days_entry_payment' in col or 'days_instalment' in col]
        if days_cols:
            df['ip_max_delay'] = df[days_cols].max(axis=1, skipna=True) - df[days_cols].min(axis=1, skipna=True)
            df['ip_avg_delay'] = df[days_cols].mean(axis=1, skipna=True)
            self.features_added.extend(['ip_max_delay', 'ip_avg_delay'])
        
        # 3. ПРИЗНАКИ РЕСТРУКТУРИЗАЦИИ
        version_cols = [col for col in df.columns if col.startswith('ip_version_')]
        if len(version_cols) > 1:
            # Флаг реструктуризации (более одной версии кредита)
            df['ip_restructured_flag'] = (df[version_cols].sum(axis=1) > 1).astype(int)
            # Разнообразие версий кредита
            df['ip_version_diversity'] = df[version_cols].gt(0).sum(axis=1)
            self.features_added.extend(['ip_restructured_flag', 'ip_version_diversity'])
        
        print(f"Гипотеза 8 (Installments): +{len(self.features_added)} фич → {self.features_added}")
        return df