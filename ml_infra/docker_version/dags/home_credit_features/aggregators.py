import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class BaseAggregator:
    """Базовый класс для агрегаторов"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.features_added = []
    
    def log(self, message: str):
        if self.verbose:
            print(message)

class BureauAggregator(BaseAggregator):
    """Агрегация данных из bureau и bureau_balance"""
    
    def __init__(self, verbose: bool = True):
        super().__init__(verbose)
        
    def process(self, bureau_df: pd.DataFrame, bureau_balance_df: pd.DataFrame) -> pd.DataFrame:
        """Основной метод обработки"""
        self.log("Processing bureau data...")
        
        # 1. One-hot encoding для категориальных колонок
        bureau = bureau_df.copy()
        bureau_balance = bureau_balance_df.copy()
        
        bureau = pd.get_dummies(bureau, 
                               columns=['credit_active', 'credit_type', 'credit_currency'], 
                               prefix=['credit_active', 'credit_type', 'credit_currency'], 
                               dummy_na=False)
        
        # 2. Обработка bureau_balance
        if 'status' in bureau_balance.columns:
            bureau_balance = pd.get_dummies(bureau_balance, 
                                          columns=['status'], 
                                          prefix='status', 
                                          dummy_na=False)
            
            # Агрегация bureau_balance
            bureau_balance_agg = bureau_balance.groupby('sk_id_bureau').agg({
                'months_balance': ['mean', 'min', 'count'],
                **{col: 'mean' for col in bureau_balance.columns if col.startswith('status_')}
            })
            bureau_balance_agg.columns = ['bb_' + '_'.join(col).strip() 
                                         for col in bureau_balance_agg.columns.values]
            bureau_balance_agg = bureau_balance_agg.reset_index()
            
            bureau = bureau.merge(bureau_balance_agg, on='sk_id_bureau', how='left').fillna(0)
        
        # 3. Заполнение пропусков
        bureau['amt_credit_sum'] = bureau['amt_credit_sum'].fillna(bureau['amt_credit_sum'].median())
        
        missing_cols = ['days_credit_enddate', 'days_enddate_fact', 'amt_credit_max_overdue', 
                       'amt_credit_sum_debt', 'amt_credit_sum_limit', 'amt_annuity']
        
        for col in missing_cols:
            if col in bureau.columns:
                bureau[f'{col}_missing'] = bureau[col].isnull().astype(int)
                bureau[col] = bureau[col].fillna(0)
        
        # 4. Агрегация по sk_id_curr
        numeric_cols = bureau.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols 
                       if col not in ['sk_id_curr', 'sk_id_bureau']]
        
        agg_dict = {}
        for col in numeric_cols:
            if col.startswith('bb_'): 
                agg_dict[col] = 'mean'
            elif any(col.startswith(prefix) for prefix in 
                    ['credit_active_', 'credit_type_', 'credit_currency_', 'status_']):
                agg_dict[col] = 'mean' 
            else:
                agg_dict[col] = ['mean', 'max', 'min']
        
        bureau_agg = bureau.groupby('sk_id_curr').agg(agg_dict)
        bureau_agg.columns = ['_'.join(col).strip() for col in bureau_agg.columns.values]
        bureau_agg = bureau_agg.reset_index()
        
        self.features_added = bureau_agg.columns.tolist()
        self.log(f"Bureau aggregator: added {len(bureau_agg.columns)} features")
        
        return bureau_agg


class PreviousApplicationAggregator(BaseAggregator):
    """Агрегация данных из previous_application"""
    
    def __init__(self, verbose: bool = True):
        super().__init__(verbose)
    
    def process(self, previous_app_df: pd.DataFrame) -> pd.DataFrame:
        """Обработка previous_application"""
        self.log("Processing previous applications...")
        
        df = previous_app_df.copy()
        
        # Заполнение пропусков 
        df['product_combination'] = df['product_combination'].fillna('Unknown')
        df['amt_credit'] = df['amt_credit'].fillna(df['amt_credit'].median())
        
        if 'name_type_suite' in df.columns:
            df['name_type_suite_missing'] = df['name_type_suite'].isnull().astype(int)
            df['name_type_suite'] = df['name_type_suite'].fillna('Unknown')
        
        missing_numeric = ['amt_annuity', 'amt_down_payment', 'amt_goods_price', 
                          'rate_down_payment', 'rate_interest_primary', 
                          'rate_interest_privileged', 'cnt_payment', 'days_first_drawing', 
                          'days_first_due', 'days_last_due_1st_version', 'days_last_due', 
                          'days_termination', 'nflag_insured_on_approval']
        
        for col in missing_numeric:
            if col in df.columns:
                df[f'{col}_missing'] = df[col].isnull().astype(int)
                df[col] = df[col].fillna(0)
        
        # Категориальные колонки для one-hot encoding
        cat_cols = ['name_contract_type', 'name_cash_loan_purpose', 'name_contract_status',
                   'name_payment_type', 'code_reject_reason', 'name_type_suite',
                   'name_client_type', 'name_goods_category', 'name_portfolio',
                   'name_product_type', 'channel_type', 'name_seller_industry',
                   'name_yield_group', 'product_combination']
        
        cat_cols = [col for col in cat_cols if col in df.columns]
        
        df = pd.get_dummies(df, columns=cat_cols, prefix=cat_cols, dummy_na=False)
        
        agg_dict = {
            'sk_id_prev': 'count',
            'amt_annuity': ['mean', 'sum'],
            'amt_application': ['mean', 'sum'],
            'amt_credit': ['mean', 'sum'],
            'amt_down_payment': ['mean', 'sum'],
            'amt_goods_price': ['mean', 'sum'],
            'days_decision': ['mean', 'min', 'max'],
            'cnt_payment': ['mean', 'sum'],
            'rate_down_payment': 'mean',
            'rate_interest_primary': 'mean',
            'rate_interest_privileged': 'mean',
            'sellerplace_area': 'mean',
            'days_first_drawing': 'mean',
            'days_first_due': 'mean',
            'days_last_due_1st_version': 'mean',
            'days_last_due': 'mean',
            'days_termination': 'mean',
            'nflag_insured_on_approval': 'mean'
        }
        
        for col in df.columns:
            if any(col.startswith(cat_col) for cat_col in cat_cols):
                agg_dict[col] = 'mean'
        
        previous_app_agg = df.groupby('sk_id_curr').agg(agg_dict)
        previous_app_agg.columns = ['_'.join(col).strip() for col in previous_app_agg.columns.values]
        
        self.features_added = previous_app_agg.columns.tolist()
        self.log(f"Previous applications aggregator: added {len(previous_app_agg.columns)} features")
        
        return previous_app_agg


class InstallmentsAggregator(BaseAggregator):
    """Агрегация данных из installments_payments"""
    
    def __init__(self, verbose: bool = True):
        super().__init__(verbose)
    
    def process(self, installments_df: pd.DataFrame) -> pd.DataFrame:
        """Обработка installments_payments"""
        self.log("Processing installments payments...")
        
        df = installments_df.copy()
        
        # Заполнение пропусков
        missing_cols = ['days_entry_payment', 'amt_payment']
        for col in missing_cols:
            if col in df.columns:
                df[f'{col}_missing'] = df[col].isnull().astype(int)
                df[col] = df[col].fillna(0)
        
        if 'num_instalment_version' in df.columns:
            df = pd.get_dummies(df, columns=['num_instalment_version'], 
                              prefix='version', dummy_na=True)
        
        agg_dict = {
            'sk_id_prev': 'count',  
            'num_instalment_number': ['mean', 'min', 'max', 'sum'],           
            'days_instalment': ['mean', 'min', 'max'],  
            'days_entry_payment': ['mean', 'min', 'max'],  
            'amt_instalment': ['mean', 'sum', 'max', 'min'],  
            'amt_payment': ['mean', 'sum', 'max', 'min']  
        }
        
        version_cols = [col for col in df.columns if col.startswith('version')]
        for col in version_cols:
            agg_dict[col] = 'mean'
        
        installments_agg = df.groupby('sk_id_curr').agg(agg_dict)
        installments_agg.columns = ['ip_' + '_'.join(col).strip() 
                                   for col in installments_agg.columns.values]
        installments_agg = installments_agg.reset_index()
        
        self.features_added = installments_agg.columns.tolist()
        self.log(f"Installments aggregator: added {len(installments_agg.columns)} features")
        
        return installments_agg


class CreditCardAggregator(BaseAggregator):
    """Агрегация данных из credit_card_balance"""
    
    def __init__(self, verbose: bool = True):
        super().__init__(verbose)
    
    def process(self, credit_card_df: pd.DataFrame) -> pd.DataFrame:
        """Обработка credit_card_balance"""
        self.log("Processing credit card balance...")
        
        df = credit_card_df.copy()
        
        missing_cols = ['amt_drawings_atm_current', 'amt_drawings_other_current', 
                       'amt_drawings_pos_current', 'amt_inst_min_regularity', 
                       'amt_payment_current', 'cnt_drawings_atm_current',
                       'cnt_drawings_other_current', 'cnt_drawings_pos_current', 
                       'cnt_instalment_mature_cum']
        
        for col in missing_cols:
            if col in df.columns:
                df[f'{col}_missing'] = df[col].isnull().astype(int)
                df[col] = df[col].fillna(0)
        
        if 'name_contract_status' in df.columns:
            df = pd.get_dummies(df, columns=['name_contract_status'], 
                              prefix='name_contract_status', dummy_na=False)
        
        agg_dict = {
            'sk_id_prev': 'count',
            'months_balance': ['mean', 'min', 'max'],
            'amt_balance': ['mean', 'max', 'min', 'sum'],
            'amt_credit_limit_actual': ['mean', 'max', 'min'],
            'amt_drawings_atm_current': ['mean', 'sum'],
            'amt_drawings_current': ['mean', 'sum'],
            'amt_drawings_other_current': ['mean', 'sum'],
            'amt_drawings_pos_current': ['mean', 'sum'],
            'amt_inst_min_regularity': ['mean', 'max', 'min'],
            'amt_payment_current': ['mean', 'sum'],
            'amt_payment_total_current': ['mean', 'sum'],
            'amt_receivable_principal': ['mean', 'sum'],
            'amt_recivable': ['mean', 'sum'],
            'amt_total_receivable': ['mean', 'sum'],
            'cnt_drawings_atm_current': ['mean', 'sum'],
            'cnt_drawings_current': ['mean', 'sum'],
            'cnt_drawings_other_current': ['mean', 'sum'],
            'cnt_drawings_pos_current': ['mean', 'sum'],
            'cnt_instalment_mature_cum': ['mean', 'max', 'min'],
            'sk_dpd': ['max', 'mean', 'min'],
            'sk_dpd_def': ['max', 'mean', 'min']
        }
        
        status_cols = [col for col in df.columns if col.startswith('name_contract_status')]
        for col in status_cols:
            agg_dict[col] = 'mean'
        
        credit_card_agg = df.groupby('sk_id_curr').agg(agg_dict)
        credit_card_agg.columns = ['_'.join(col).strip() for col in credit_card_agg.columns.values]
        
        self.features_added = credit_card_agg.columns.tolist()
        self.log(f"Credit card aggregator: added {len(credit_card_agg.columns)} features")
        
        return credit_card_agg


class PosCashAggregator(BaseAggregator):
    """Агрегация данных из POS_CASH_balance"""
    
    def __init__(self, verbose: bool = True):
        super().__init__(verbose)
    
    def process(self, pos_cash_df: pd.DataFrame) -> pd.DataFrame:
        """Обработка POS_CASH_balance"""
        self.log("Processing POS cash balance...")
        
        df = pos_cash_df.copy()
        
        df['cnt_instalment'] = df['cnt_instalment'].fillna(df['cnt_instalment'].median())
        
        if 'cnt_instalment_future' in df.columns:
            df['cnt_instalment_future_missing'] = df['cnt_instalment_future'].isnull().astype(int)
            df['cnt_instalment_future'] = df['cnt_instalment_future'].fillna(0)
        
        if 'name_contract_status' in df.columns:
            df = pd.get_dummies(df, columns=['name_contract_status'], 
                              prefix='name_contract_status', dummy_na=False)
        
        agg_dict = {
            'sk_id_prev': 'count',
            'months_balance': ['mean', 'min', 'max'],
            'cnt_instalment': 'mean',
            'cnt_instalment_future': 'mean',
            'sk_dpd': ['max', 'mean'],
            'sk_dpd_def': ['max', 'mean']
        }
        
        status_cols = [col for col in df.columns if col.startswith('name_contract_status')]
        for col in status_cols:
            agg_dict[col] = 'mean'
        
        pos_cash_agg = df.groupby('sk_id_curr').agg(agg_dict)
        pos_cash_agg.columns = ['_'.join(col).strip() for col in pos_cash_agg.columns.values]
        
        self.features_added = pos_cash_agg.columns.tolist()
        self.log(f"POS cash aggregator: added {len(pos_cash_agg.columns)} features")
        
        return pos_cash_agg