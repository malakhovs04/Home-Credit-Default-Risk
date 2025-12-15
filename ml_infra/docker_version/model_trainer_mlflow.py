import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import os
import sys
import mlflow
import mlflow.sklearn
from datetime import datetime
import warnings
import json
import tempfile
import seaborn as sns
from sklearn.metrics import confusion_matrix

warnings.filterwarnings('ignore')

# Добавляем путь для импорта
sys.path.append('/app')

plt.rcParams['axes.grid'] = True
plt.rcParams['figure.figsize'] = (10, 6)

class ModelTrainerMLflow:
    def __init__(self, model, experiment_name="home-credit-default-risk", run_name=None):
        self.model = model
        self.name = experiment_name
        self.run_name = run_name or f"{type(model).__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.val_auc = None
        self.train_auc = None
        self.best_threshold = 0.5
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.f1 = None
        self.pr_auc = None
        self.feature_importance = None
        self.best_iteration = None
        
        self.mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(self.name)

    def fit_with_mlflow(self, X_train, y_train, X_val=None, y_val=None, 
                       log_params=True, log_metrics=True, log_model=True, log_artifacts=True):
        
        print(f"🚀 Начинаем обучение модели: {type(self.model).__name__}")
        print(f"📊 Эксперимент: {self.name}")
        print(f"📈 Run: {self.run_name}")
        print(f"🔗 MLflow Tracking URI: {self.mlflow_tracking_uri}")
        
        with mlflow.start_run(run_name=self.run_name):
            if log_params:
                try:
                    params = self.model.get_params()
                    mlflow.log_params(params)
                    print(f"✅ Логированы параметры: {len(params)} шт.")
                except Exception as e:
                    print(f"⚠️ Не удалось залогировать параметры: {e}")
            
            if X_val is None or y_val is None:
                X_train_split, X_val, y_train_split, y_val = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
                )
                print(f"📊 Разделение данных: {X_train_split.shape[0]} train, {X_val.shape[0]} val")
            else:
                X_train_split, y_train_split = X_train, y_train
            
            print("🎯 Обучение модели...")
            model_name = str(type(self.model)).lower()
            
            # УНИВЕРСАЛЬНЫЙ БЛОК ОБУЧЕНИЯ ДЛЯ ЛЮБЫХ МОДЕЛЕЙ
            try:
                if "catboost" in model_name:
                    # CatBoost: eval_set как кортеж
                    try:
                        # Сначала пробуем с eval_set
                        self.model.fit(
                            X_train_split, y_train_split,
                            eval_set=(X_val, y_val),
                            early_stopping_rounds=10,
                            verbose=False
                        )
                        self.best_iteration = getattr(self.model, 'best_iteration_', 
                                                     getattr(self.model, 'get_best_iteration', lambda: None)())
                    except:
                        # Если не работает с eval_set, пробуем без
                        self.model.fit(X_train_split, y_train_split, verbose=False)
                        
                elif "lgbm" in model_name:
                    # LightGBM: eval_set как список
                    try:
                        import lightgbm as lgb
                        self.model.fit(
                            X_train_split, y_train_split,
                            eval_set=[(X_val, y_val)],
                            callbacks=[lgb.early_stopping(10)],
                            verbose=False
                        )
                        self.best_iteration = getattr(self.model, 'best_iteration_', None)
                    except:
                        self.model.fit(X_train_split, y_train_split)
                        
                elif "xgb" in model_name:
                    # XGBoost: eval_set как список
                    try:
                        self.model.fit(
                            X_train_split, y_train_split,
                            eval_set=[(X_val, y_val)],
                            early_stopping_rounds=10,
                            verbose=False
                        )
                        self.best_iteration = getattr(self.model, 'best_iteration', None)
                    except:
                        self.model.fit(X_train_split, y_train_split)
                else:
                    # Любые другие модели (LogisticRegression, RandomForest и т.д.)
                    self.model.fit(X_train_split, y_train_split)
                    
                print(f"✅ Модель успешно обучена")
                
            except Exception as e:
                print(f"❌ Критическая ошибка при обучении: {e}")
                raise
            
            # Получаем предсказания
            try:
                if hasattr(self.model, "predict_proba"):
                    train_proba = self.model.predict_proba(X_train_split)[:, 1]
                    val_proba = self.model.predict_proba(X_val)[:, 1]
                else:
                    train_proba = self.model.predict(X_train_split)
                    val_proba = self.model.predict(X_val)
                    print("⚠️ Модель не поддерживает predict_proba, используем predict")
            except Exception as e:
                print(f"❌ Ошибка при получении предсказаний: {e}")
                raise
            
            # Вычисляем метрики
            try:
                self.train_auc = roc_auc_score(y_train_split, train_proba)
                self.val_auc = roc_auc_score(y_val, val_proba)
                
                precision, recall, _ = precision_recall_curve(y_val, val_proba)
                self.pr_auc = auc(recall, precision)
                
                fpr, tpr, thresholds = roc_curve(y_val, val_proba)
                if len(thresholds) > 0:
                    J = tpr - fpr
                    best_idx = np.argmax(J)
                    self.best_threshold = thresholds[best_idx]
                else:
                    self.best_threshold = 0.5
                    
                val_pred_optimal = (val_proba >= self.best_threshold).astype(int)
                
                self.accuracy = accuracy_score(y_val, val_pred_optimal)
                self.precision = precision_score(y_val, val_pred_optimal, zero_division=0)
                self.recall = recall_score(y_val, val_pred_optimal, zero_division=0)
                self.f1 = f1_score(y_val, val_pred_optimal, zero_division=0)
                
                print(f"📊 Метрики вычислены: ROC-AUC={self.val_auc:.4f}, Accuracy={self.accuracy:.4f}")
                
            except Exception as e:
                print(f"⚠️ Ошибка вычисления метрик: {e}")
                # Устанавливаем значения по умолчанию
                self.train_auc = 0.5
                self.val_auc = 0.5
                self.pr_auc = 0.5
                self.accuracy = 0.5
                self.precision = 0.5
                self.recall = 0.5
                self.f1 = 0.5
            
            if log_metrics:
                metrics = {
                    "val_roc_auc": float(self.val_auc),
                    "train_roc_auc": float(self.train_auc),
                    "val_pr_auc": float(self.pr_auc),
                    "accuracy": float(self.accuracy),
                    "precision": float(self.precision),
                    "recall": float(self.recall),
                    "f1_score": float(self.f1),
                    "best_threshold": float(self.best_threshold),
                    "train_samples": int(len(X_train_split)),
                    "val_samples": int(len(X_val))
                }
                
                if self.best_iteration is not None:
                    metrics["best_iteration"] = int(self.best_iteration)
                
                try:
                    mlflow.log_metrics(metrics)
                    print(f"✅ Логированы метрики: {len(metrics)} шт.")
                except Exception as e:
                    print(f"⚠️ Ошибка логирования метрик: {e}")
            
            if log_model:
                try:
                    mlflow.sklearn.log_model(
                        sk_model=self.model,
                        artifact_path="model",
                        registered_model_name=f"home_credit_{type(self.model).__name__.lower()}"
                    )
                    print("✅ Модель залогирована в MLflow")
                except Exception as e:
                    print(f"⚠️ Не удалось залогировать модель: {e}")
            
            if log_artifacts:
                self._create_and_log_artifacts(y_val, val_proba, X_train)
            
            self._print_results()
            
            return self
    
    def _create_and_log_artifacts(self, y_val, val_proba, X_train):
        artifacts_dir = "/tmp/mlflow_artifacts"
        os.makedirs(artifacts_dir, exist_ok=True)
        
        try:
            roc_pr_fig = self._plot_roc_pr_curves(y_val, val_proba)
            roc_pr_path = os.path.join(artifacts_dir, "roc_pr_curves.png")
            roc_pr_fig.savefig(roc_pr_path, dpi=150, bbox_inches='tight')
            plt.close(roc_pr_fig)
            mlflow.log_artifact(roc_pr_path, "plots")
            print("✅ ROC/PR кривые сохранены")
        except Exception as e:
            print(f"⚠️ Ошибка при создании ROC/PR кривых: {e}")
        
        try:
            if hasattr(self.model, "feature_importances_"):
                feature_importance = self._create_feature_importance_df(X_train)
                importance_path = os.path.join(artifacts_dir, "feature_importance.csv")
                feature_importance.to_csv(importance_path, index=False)
                mlflow.log_artifact(importance_path, "feature_importance")
                
                top_features = feature_importance.head(20)['feature'].tolist()
                mlflow.log_param("top_20_features", json.dumps(top_features))
                print("✅ Feature importance сохранен")
        except Exception as e:
            print(f"⚠️ Ошибка при создании feature importance: {e}")
        
        try:
            val_pred_optimal = (val_proba >= self.best_threshold).astype(int)
            cm = confusion_matrix(y_val, val_pred_optimal)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
            
            cm_path = os.path.join(artifacts_dir, "confusion_matrix.png")
            fig.savefig(cm_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            mlflow.log_artifact(cm_path, "plots")
            print("✅ Confusion matrix сохранена")
        except Exception as e:
            print(f"⚠️ Ошибка при создании confusion matrix: {e}")
    
    def _plot_roc_pr_curves(self, y_val, val_proba):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        try:
            fpr, tpr, _ = roc_curve(y_val, val_proba)
            roc_auc = auc(fpr, tpr)
            ax1.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}', color='blue', linewidth=2)
            ax1.plot([0, 1], [0, 1], 'k--', linewidth=1)
            ax1.set_xlabel('False Positive Rate')
            ax1.set_ylabel('True Positive Rate')
            ax1.set_title('ROC Curve')
            ax1.legend()
            ax1.grid(True)
        except Exception as e:
            ax1.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center')
            ax1.set_title('ROC Curve (Error)')
        
        try:
            precision, recall, _ = precision_recall_curve(y_val, val_proba)
            pr_auc = auc(recall, precision)
            ax2.plot(recall, precision, label=f'AUC = {pr_auc:.4f}', color='green', linewidth=2)
            ax2.set_xlabel('Recall')
            ax2.set_ylabel('Precision')
            ax2.set_title('Precision-Recall Curve')
            ax2.legend()
            ax2.grid(True)
        except Exception as e:
            ax2.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center')
            ax2.set_title('PR Curve (Error)')
        
        plt.suptitle(f"{self.name} | Val ROC-AUC = {self.val_auc:.4f}" if self.val_auc else f"{self.name}")
        plt.tight_layout()
        
        return fig
    
    def _create_feature_importance_df(self, X_train):
        if hasattr(self.model, "feature_importances_"):
            imp = self.model.feature_importances_
        elif hasattr(self.model, "get_feature_importance"):
            imp = self.model.get_feature_importance()
        else:
            imp = np.zeros(X_train.shape[1])
        
        feature_importance = pd.DataFrame({
            'feature': list(X_train.columns),
            'importance': imp
        }).sort_values('importance', ascending=False).reset_index(drop=True)
        
        return feature_importance
    
    def _print_results(self):
        print(f"\n{'='*60}")
        print(f"🎯 РЕЗУЛЬТАТЫ ОБУЧЕНИЯ — {self.name}")
        print(f"{'='*60}")
        print(f"Model: {type(self.model).__name__}")
        print(f"Run: {self.run_name}")
        print(f"{'-'*60}")
        
        metrics_info = [
            ("Val ROC-AUC", self.val_auc, ".5f"),
            ("Val PR-AUC", self.pr_auc, ".5f"),
            ("Train ROC-AUC", self.train_auc, ".5f"),
            ("Accuracy", self.accuracy, ".4f"),
            ("Precision", self.precision, ".4f"),
            ("Recall", self.recall, ".4f"),
            ("F1-score", self.f1, ".4f"),
        ]
        
        for name, value, fmt in metrics_info:
            if value is not None:
                print(f"{name:<20}: {value:{fmt}}")
            else:
                print(f"{name:<20}: N/A")
                
        print(f"Лучший порог      : {self.best_threshold:.4f}")
        
        if self.best_iteration:
            print(f"Лучшая итерация   : {self.best_iteration}")
        
        print(f"{'='*60}")
        print(f"✅ Результаты залогированы в MLflow!")
        print(f"🔗 Откройте: {self.mlflow_tracking_uri}")
        print(f"{'='*60}")