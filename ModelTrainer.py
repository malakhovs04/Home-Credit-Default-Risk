import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score,
)
from sklearn.model_selection import train_test_split
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['axes.grid'] = True
plt.rcParams['figure.figsize'] = (10, 6)

class ModelTrainer:
    def __init__(self, model, experiment_name="baseline"):
        self.model = model
        self.name = experiment_name
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Метрики
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

    def fit(self, X_train, y_train, X_val=None, y_val=None, cat_features=None):   
        print(f"Модель: {type(self.model).__name__}")

        if X_val is None or y_val is None:
            X_train_split, X_val, y_train_split, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
            print(f"Сплит: {X_train_split.shape[0]} train, {X_val.shape[0]} val")
        else:
            X_train_split, y_train_split = X_train, y_train

        model_name = str(type(self.model)).lower()

        # Обучение с early stopping
        if "catboost" in model_name:
            self.model.fit(X_train_split, y_train_split,
                          eval_set=(X_val, y_val),
                          
                          verbose=100, early_stopping_rounds=100, use_best_model=True)
            self.best_iteration = self.model.get_best_iteration()
        elif "lgbm" in model_name:
            self.model.fit(X_train_split, y_train_split,
                          eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=False)
            self.best_iteration = self.model.best_iteration_
        elif "xgb" in model_name:
            self.model.fit(X_train_split, y_train_split,
                          eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=False)
            self.best_iteration = self.model.best_iteration_
        else:
            self.model.fit(X_train_split, y_train_split)

        # Предсказания
        train_proba = self.model.predict_proba(X_train_split)[:, 1]
        val_proba = self.model.predict_proba(X_val)[:, 1]
        val_pred = (val_proba >= self.best_threshold).astype(int)

        # ROC-AUC
        self.train_auc = roc_auc_score(y_train_split, train_proba)
        self.val_auc = roc_auc_score(y_val, val_proba)

        # PR-AUC
        precision, recall, _ = precision_recall_curve(y_val, val_proba)
        self.pr_auc = auc(recall, precision)

        # Лучший порог 
        fpr, tpr, thresholds = roc_curve(y_val, val_proba)
        J = tpr - fpr
        best_idx = np.argmax(J)
        self.best_threshold = thresholds[best_idx]
        val_pred_optimal = (val_proba >= self.best_threshold).astype(int)

        # Метрики с лучшим порогом
        self.accuracy = accuracy_score(y_val, val_pred_optimal)
        self.precision = precision_score(y_val, val_pred_optimal)
        self.recall = recall_score(y_val, val_pred_optimal)
        self.f1 = f1_score(y_val, val_pred_optimal)

        # Вывод всех метрик
        print(f"РЕЗУЛЬТАТЫ — {self.name}")
        print(f"Val ROC-AUC       : {self.val_auc:.5f}")
        print(f"Val PR-AUC        : {self.pr_auc:.5f}")
        print(f"Train ROC-AUC     : {self.train_auc:.5f}")
        print(f"Лучший порог      : {self.best_threshold:.4f}")
        print(f"Accuracy          : {self.accuracy:.4f}")
        print(f"Precision         : {self.precision:.4f}")
        print(f"Recall            : {self.recall:.4f}")
        print(f"F1-score          : {self.f1:.4f}")
        if self.best_iteration:
            print(f"Лучшая итерация   : {self.best_iteration}")
        print(f"{'='*50}")

        # Feature importance
        if hasattr(self.model, "feature_importances_"):
            imp = self.model.feature_importances_
        elif hasattr(self.model, "get_feature_importance"):
            imp = self.model.get_feature_importance()
        else:
            imp = None

        if imp is not None:
            self.feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': imp
            }).sort_values('importance', ascending=False).reset_index(drop=True)

        return self

    def plot_curves(self, y_val, val_proba):
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        fpr, tpr, _ = roc_curve(y_val, val_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}', color='blue', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        precision, recall, _ = precision_recall_curve(y_val, val_proba)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f'AUC = {pr_auc:.4f}', color='green', linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)

        auc_text = f"Val ROC-AUC = {self.val_auc:.5f}" if self.val_auc is not None else "Val ROC-AUC = N/A"
        plt.suptitle(f"{self.name} | {auc_text}")
        plt.tight_layout()
        plt.show()

    def plot_importance(self, top_n=20):
        if self.feature_importance is None:
            print("Нет feature importance")
            return

        top = self.feature_importance.head(top_n)
        plt.figure(figsize=(10, 8))
        plt.barh(range(top_n-1, -1, -1), top['importance'], color='skyblue')
        plt.yticks(range(top_n-1, -1, -1), top['feature'])
        plt.xlabel('Importance')
        plt.title(f"Top {top_n} Features — {self.name}")
        plt.tight_layout()
        plt.show()

    def predict_test(self, X_test):
        proba = self.model.predict_proba(X_test)[:, 1]
        pred = (proba >= self.best_threshold).astype(int)
        return proba, pred

    def make_submission(self, test_ids, test_proba):
        sub = pd.DataFrame({
            'SK_ID_CURR': test_ids,
            'TARGET': test_proba
        })
        filename = f"{self.results_dir}/submission_{self.name}.csv"
        sub.to_csv(filename, index=False)