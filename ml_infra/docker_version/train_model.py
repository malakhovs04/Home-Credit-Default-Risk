import os
import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Docker
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
)
from sqlalchemy import create_engine, text
import mlflow
from datetime import datetime

warnings.filterwarnings('ignore')

# Add paths for custom imports
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/home_credit_features')

print(f"Python path: {sys.path}")

# Import ModelTrainerMLflow
try:
    from model_trainer_mlflow import ModelTrainerMLflow
    print("ModelTrainerMLflow imported successfully")
except ImportError as e:
    print(f"Error importing ModelTrainerMLflow: {e}")
    sys.exit(1)


class CatBoostTrainer:
    def __init__(self, table_name: str = "test_train_data_cleaned", experiment_name: str = "home-credit-default-risk"):
        self.table_name = table_name
        self.experiment_name = experiment_name
        self.model = None
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.feature_names = None

    def load_data(self) -> pd.DataFrame:
        """Load data from PostgreSQL with fallback to test data"""
        print(f"Loading data from table: {self.table_name}")

        db_host = os.getenv('DB_HOST', 'host.docker.internal')
        db_user = os.getenv('DB_USER', 'postgres')
        db_password = os.getenv('DB_PASSWORD', '12345')
        db_name = os.getenv('DB_NAME', 'postgres')
        db_port = os.getenv('DB_PORT', '5432')

        connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

        try:
            engine = create_engine(connection_string)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            print(f"Database connection successful: {db_host}:{db_port}/{db_name}")

            row_limit = int(os.getenv('ROW_LIMIT', '100'))
            query = f"SELECT * FROM {self.table_name} LIMIT {row_limit}"
            print(f"Executing query: {query}")

            df = pd.read_sql(query, engine)
            print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
            return df

        except Exception as e:
            print(f"Database error: {e}")
            print("Falling back to generated test data")
            return self._generate_test_data()

    def _generate_test_data(self) -> pd.DataFrame:
        """Generate synthetic test data"""
        n_samples = 100
        n_features = 95

        data = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        data['target'] = np.random.randint(0, 2, n_samples)
        data['category_1'] = np.random.choice(['A', 'B', 'C'], n_samples)
        data['category_2'] = np.random.choice(['X', 'Y'], n_samples)

        print(f"Generated test data: {data.shape}")
        return data

    def preprocess_data(self, data: pd.DataFrame):
        """Split data and apply One-Hot Encoding"""
        target_col = "target"
        if target_col not in data.columns:
            print(f"Warning: '{target_col}' not found. Using last column as target")
            target_col = data.columns[-1]

        X = data.drop(columns=[target_col])
        y = data[target_col]

        print(f"Target column: {target_col}")
        print(f"Data shape: X={X.shape}, y={y.shape}")

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"Split: train={X_train.shape}, val={X_val.shape}")

        # One-Hot Encoding
        cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        print(f"Found {len(cat_cols)} categorical features")

        if cat_cols:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
            X_train_cat = ohe.fit_transform(X_train[cat_cols])
            X_val_cat = ohe.transform(X_val[cat_cols])

            ohe_cols = ohe.get_feature_names_out(cat_cols)

            X_train_cat_df = pd.DataFrame(X_train_cat, columns=ohe_cols, index=X_train.index)
            X_val_cat_df = pd.DataFrame(X_val_cat, columns=ohe_cols, index=X_val.index)

            num_cols = [col for col in X_train.columns if col not in cat_cols]
            X_train_numeric = X_train[num_cols]
            X_val_numeric = X_val[num_cols]

            X_train = pd.concat([X_train_numeric, X_train_cat_df], axis=1)
            X_val = pd.concat([X_val_numeric, X_val_cat_df], axis=1)

            print(f"One-Hot Encoding applied. Final features: {X_train.shape[1]}")

        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.feature_names = X_train.columns.tolist()

    def create_model(self):
        """Create CatBoost model"""
        print("Creating CatBoostClassifier")
        try:
            from catboost import CatBoostClassifier

            self.model = CatBoostClassifier(
                iterations=100,
                learning_rate=0.05,
                depth=6,
                loss_function='Logloss',
                verbose=False,
                random_seed=42,
                task_type='CPU'
            )
            print("CatBoost model created")
        except Exception as e:
            print(f"Error creating CatBoost model: {e}")
            raise

    def train_and_log(self):
        """Train model and log everything to MLflow"""
        print("Starting training and logging to MLflow")

        # Set tracking URI
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(tracking_uri)
        print(f"MLflow tracking URI: {tracking_uri}")

        try:
            trainer = ModelTrainerMLflow(
                model=self.model,
                experiment_name=self.experiment_name,
                run_name=f"catboost_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

            trainer.fit_with_mlflow(
                X_train=self.X_train,
                y_train=self.y_train,
                X_val=self.X_val,
                y_val=self.y_val,
                log_params=True,
                log_metrics=True,
                log_model=True,
                log_artifacts=True
            )

            print("Training and MLflow logging completed successfully")

        except Exception as e:
            print(f"Error during training or logging: {e}")
            import traceback
            traceback.print_exc()
            raise

    def run(self):
        """Full training pipeline"""
        print("Starting CatBoost training pipeline")
        print("=" * 60)

        data = self.load_data()
        self.preprocess_data(data)
        self.create_model()

        self.model.fit(self.X_train, self.y_train)

        self.train_and_log()


# Запуск при прямом выполнении файла (для Docker CMD)
trainer = CatBoostTrainer(
    table_name=os.getenv("TABLE_NAME", "test_train_data_cleaned"),
    experiment_name=os.getenv("EXPERIMENT_NAME", "home-credit-default-risk")
)
trainer.run()