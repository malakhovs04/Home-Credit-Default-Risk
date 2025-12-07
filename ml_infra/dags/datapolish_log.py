import pandas as pd
import numpy as np
from typing import Optional
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings('ignore')

class DataPolish(BaseEstimator, TransformerMixin):
    def __init__(self, 
                max_na_ratio: float = 0.7,
                low_variance_quantile: float = 0.01,
                corr_threshold: float = 0.95,
                handle_outliers: bool = True,
                outlier_method = 'clip',
                log_transform_skew: float = 2.0,
                scale_numeric: bool = False,
                random_state: int = 42,
                verbose: bool = True):
        
        self.max_na_ratio = max_na_ratio
        self.low_variance_quantile = low_variance_quantile
        self.corr_threshold = corr_threshold
        self.handle_outliers = handle_outliers
        self.outlier_method = outlier_method
        self.log_transform_skew = log_transform_skew
        self.scale_numeric = scale_numeric
        self.random_state = random_state
        self.verbose = verbose

        self.dropped_columns_: list[str] = []
        self.medians_: Optional[pd.Series] = None
        self.modes_: Optional[pd.Series] = None
        self.scaler_: Optional[StandardScaler] = None
        self.skewed_columns_: list[str] = []
        self.columns_to_keep_: Optional[list[str]] = None
        self.categorical_columns_ = [] 
    
    def log(self, message: str):
        if self.verbose:
            print(f"[DataPolish] {message}")
    
    def _remove_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        self.log("Удаление низкокачественных колонок")
        df = df.copy()
        cols_to_drop = set()
        initial_cols = df.shape[1]

        # 1. Слишком много пропусков
        na_ratio = df.isnull().mean()
        high_na_cols = na_ratio[na_ratio > self.max_na_ratio].index.tolist()
        cols_to_drop.update(high_na_cols)
        if high_na_cols:
            self.log(f"  Удалено {len(high_na_cols)} колонок с >{self.max_na_ratio*100}% пропусков")

        # 2. Константные колонки
        const_cols = [col for col in df.columns if df[col].nunique() <= 1]
        cols_to_drop.update(const_cols)
        if const_cols:
            self.log(f"  Удалено {len(const_cols)} константных колонок")

        # 3. Численные колонки с нулевой дисперсией
        num_cols = df.select_dtypes(include=np.number).columns
        zero_var_cols = [col for col in num_cols if np.isclose(df[col].var(), 0)]
        cols_to_drop.update(zero_var_cols)
        if zero_var_cols:
            self.log(f"  Удалено {len(zero_var_cols)} колонок с нулевой дисперсией")

        # 4. Колонки с очень низкой дисперсией
        if self.low_variance_quantile > 0 and len(num_cols) > 10:
            variances = df[num_cols].var()
            if not variances.empty:
                threshold = variances.quantile(self.low_variance_quantile)
                low_var_cols = variances[variances <= threshold].index.tolist()
                cols_to_drop.update(low_var_cols)
                if low_var_cols:
                    self.log(f"  Удалено {len(low_var_cols)} колонок с низкой дисперсией")

        df_clean = df.drop(columns=cols_to_drop)
        self.dropped_columns_.extend(cols_to_drop)

        removed_count = initial_cols - df_clean.shape[1]
        self.log(f"  Всего удалено: {removed_count} колонок, осталось: {df_clean.shape[1]}")
        
        return df_clean
    
    def _remove_highly_correlated(self, df: pd.DataFrame) -> pd.DataFrame:
        self.log("Удаление высококоррелированных колонок")
        df = df.copy()
        num_cols = df.select_dtypes(include=np.number).columns

        if len(num_cols) < 2 or self.corr_threshold >= 1.0:
            return df
        
        corr_matrix = df[num_cols].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > self.corr_threshold)]

        df_clean = df.drop(columns=to_drop)
        self.dropped_columns_.extend(to_drop)

        if to_drop:
            self.log(f"  Удалено {len(to_drop)} высококоррелированных колонок (> {self.corr_threshold})")

        return df_clean
    
    def _handle_missing_and_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        self.log("Обработка пропусков и выбросов")
        df = df.copy()
        num_cols = df.select_dtypes(include=np.number).columns
        cat_cols = df.select_dtypes(exclude=np.number).columns
         
        # Заполнение пропусков
        self.medians_ = df[num_cols].median()
        self.modes_ = df[cat_cols].mode().iloc[0] if len(cat_cols) > 0 else None

        df[num_cols] = df[num_cols].fillna(self.medians_)
        if self.modes_ is not None:
            df[cat_cols] = df[cat_cols].fillna(self.modes_)
        
        self.log(f"  Заполнено пропусков: {len(num_cols)} числовых, {len(cat_cols)} категориальных")

        # Обработка выбросов (IQR метод)
        if self.handle_outliers and self.outlier_method == "clip":
            self.log("  Обрезка выбросов методом IQR")
            for col in num_cols:
                Q1 = self.medians_[col] - 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25))
                Q3 = self.medians_[col] + 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25))
                lower = Q1 - 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25))
                upper = Q3 + 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25))
                df[col] = df[col].clip(lower, upper)
        
        # Логарифмирование сильно скошенных признаков
        if self.log_transform_skew > 0:
            skew = df[num_cols].skew().abs()
            self.skewed_columns_ = skew[skew > self.log_transform_skew].index.tolist()

            for col in self.skewed_columns_:
                offset = 0
                if df[col].min() <= 0:
                    offset = abs(df[col].min()) + 1
                df[col] = np.log1p(df[col] + offset)

            if self.skewed_columns_:
                self.log(f"  Логарифмировано {len(self.skewed_columns_)} сильно скошенных колонок")
        
        return df

    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.scale_numeric:
            return df
            
        num_cols = df.select_dtypes(include=np.number).columns
        if len(num_cols) == 0:
            return df
        
        self.log(f"Стандартизация {len(num_cols)} числовых признаков")
        self.scaler_ = StandardScaler()
        df[num_cols] = self.scaler_.fit_transform(df[num_cols])
        
        return df
    
    def fit(self, df: pd.DataFrame, y=None) -> 'DataPolish':
        self.log(f"Обучение DataPolish на данных размером: {df.shape}")
        
        df = df.copy()

        df = self._remove_columns(df)
        df = self._remove_highly_correlated(df)
        df = self._handle_missing_and_outliers(df)
        df = self._scale_features(df)

        self.columns_to_keep_ = df.columns.tolist()
        self.categorical_columns_ = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        self.log(f"Всего удалено {len(self.dropped_columns_)} колонок")
 
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:        
        missing_cols = [col for col in self.columns_to_keep_ if col not in df.columns]
        if missing_cols:
            self.log(f"Предупреждение: {len(missing_cols)} ожидаемых колонок отсутствуют")
        
        df = df.copy()
        df = df[[col for col in self.columns_to_keep_ if col in df.columns]].copy()

        num_cols = df.select_dtypes(include=np.number).columns
        cat_cols = df.select_dtypes(exclude=np.number).columns

        # Заполнение пропусков
        if self.medians_ is not None:
            for col in num_cols:
                if col in self.medians_.index:
                    df[col] = df[col].fillna(self.medians_[col])
        
        if self.modes_ is not None and len(cat_cols) > 0:
            for col in cat_cols:
                if col in self.modes_.index:
                    df[col] = df[col].fillna(self.modes_[col])

        # Выбросы 
        if self.handle_outliers and len(num_cols) > 0:
            Q1 = df[num_cols].quantile(0.25)
            Q3 = df[num_cols].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            df[num_cols] = df[num_cols].clip(lower_bound, upper_bound, axis=1)

        # Логарифмирование
        for col in self.skewed_columns_:
            if col in df.columns:
                offset = 0
                if df[col].min() <= 0:
                    offset = abs(df[col].min()) + 1
                df[col] = np.log1p(df[col] + offset)

        # Масштабирование
        if self.scale_numeric and self.scaler_ is not None:
            df[num_cols] = self.scaler_.transform(df[num_cols])
        
        self.log(f"Преобразование завершено. Итоговый размер: {df.shape}")
        return df

    def get_dropped_columns(self) -> list[str]:
        return self.dropped_columns_