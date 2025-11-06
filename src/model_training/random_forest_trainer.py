"""
ランダムフォレスト訓練クラス
分類器の訓練、評価、保存を行う
"""

import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils.config_manager import ConfigManager
from ..utils.timestamp_manager import TimestampManager


class RandomForestTrainer:
    """ランダムフォレスト訓練クラス"""
    
    def __init__(self, config: Dict[str, Any], task_type: str = 'impressionist', 
                 features_file: Optional[str] = None,
                 timestamp_manager: Optional['TimestampManager'] = None):
        """
        初期化
        
        Args:
            config: 設定辞書
            task_type: 分類タスクタイプ ('impressionist': 印象派 vs 非印象派, 'authenticity': 本物 vs フェイク)
            features_file: 特徴量ファイルのパス（Noneの場合は自動検出）
            timestamp_manager: タイムスタンプ管理オブジェクト（Noneの場合は新規作成）
                             段階的実行で同じタイムスタンプを使用する場合に指定
        """
        from ..utils.timestamp_manager import TimestampManager
        
        self.config = config
        self.data_config = config['data']
        self.classification_config = config['classification']
        self.rf_config = self.classification_config['random_forest']
        self.task_type = task_type  # 'impressionist' or 'authenticity'
        self.features_file = features_file  # カスタム特徴量ファイルパス
        
        self.logger = logging.getLogger(__name__)
        
        # 生データは固定ディレクトリから読み込み、分析結果はタイムスタンプ付きディレクトリに保存
        self.base_output_dir = Path(self.data_config['output_dir'])
        
        # タイムスタンプ管理（共有または新規作成）
        if timestamp_manager is not None:
            self.timestamp_manager = timestamp_manager
        else:
            self.timestamp_manager = TimestampManager(config)
        self.output_dir = self.timestamp_manager.get_output_dir()
        self.models_dir = self.timestamp_manager.get_models_dir()
        self.results_dir = self.timestamp_manager.get_results_dir()
        self.viz_dir = self.timestamp_manager.get_visualizations_dir()
        
        # ディレクトリ作成
        self.timestamp_manager.create_directories()
        
        # ファイルパス
        self.model_file = self.timestamp_manager.get_model_file()
        self.results_file = self.timestamp_manager.get_results_file()
        
        # モデルとエンコーダー
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.scaler = StandardScaler()
    
    def load_features(self) -> pd.DataFrame:
        """特徴量データを読み込み"""
        # カスタムファイルパスが指定されている場合
        if self.features_file is not None:
            features_file = Path(self.features_file)
            if not features_file.exists():
                raise FileNotFoundError(f"特徴量ファイルが見つかりません: {features_file}")
            df = pd.read_csv(features_file)
            self.logger.info(f"特徴量データ読み込み完了: {len(df)}件 (ファイル: {features_file})")
            return df
        
        # タスクタイプに応じて特徴量ファイルを決定
        if self.task_type == 'authenticity':
            # WikiArt_VLMデータセットの特徴量ファイルを検索
            features_dir = self.timestamp_manager.get_features_dir()
            wikiart_files = list(features_dir.glob('wikiart_vlm_features_*.csv'))
            if wikiart_files:
                features_file = wikiart_files[0]  # 最初に見つかったファイルを使用
                self.logger.info(f"WikiArt_VLM特徴量ファイルを検出: {features_file}")
            else:
                # タイムスタンプ管理のデフォルトファイルを使用
                features_file = self.timestamp_manager.get_features_file()
        else:
            # 従来のMet APIデータセット
            features_file = self.timestamp_manager.get_features_file()
        
        if not features_file.with_suffix('.csv').exists() and not isinstance(features_file, Path):
            # Path型でない場合は再度試行
            features_file = Path(features_file)
        
        if not features_file.with_suffix('.csv').exists():
            raise FileNotFoundError(f"特徴量ファイルが見つかりません: {features_file}")
        
        df = pd.read_csv(features_file.with_suffix('.csv'))
        self.logger.info(f"特徴量データ読み込み完了: {len(df)}件 (タスク: {self.task_type})")
        return df
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        訓練データを準備
        
        Args:
            df: 特徴量データフレーム
            
        Returns:
            X: 特徴量配列
            y: ラベル配列
            feature_names: 特徴量名のリスト
        """
        # 数値特徴量のみを抽出
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        # 除外するカラム（ID系とラベル）
        exclude_columns = ['object_id', 'image_id', 'label']
        feature_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        # 特徴量とラベルを分離
        X = df[feature_columns].values
        self.feature_columns = feature_columns
        
        # タスクタイプに応じてラベルを準備
        if self.task_type == 'authenticity':
            # 本物 vs フェイク分類（WikiArt_VLMデータセット）
            if 'label' not in df.columns:
                raise ValueError("WikiArt_VLMデータセットには'label'カラムが必要です (0: 本物, 1: フェイク)")
            
            # ラベルは既に数値（0: 本物, 1: フェイク）
            y = df['label'].values
            
            self.logger.info(f"データ準備完了 (本物 vs フェイク分類):")
            self.logger.info(f"  - 特徴量数: {len(feature_columns)}")
            self.logger.info(f"  - サンプル数: {len(X)}")
            self.logger.info(f"  - 本物 (0): {np.sum(y == 0)}")
            self.logger.info(f"  - フェイク (1): {np.sum(y == 1)}")
            
        else:
            # 印象派 vs 非印象派分類（Met APIデータセット）
            # 印象派の主要アーティストを定義
            impressionist_artists = [
                'Claude Monet', 'Pierre-Auguste Renoir', 'Edgar Degas', 'Camille Pissarro',
                'Berthe Morisot', 'Alfred Sisley', 'Paul Cézanne', 'Vincent van Gogh',
                'Paul Gauguin', 'Henri de Toulouse-Lautrec', 'Mary Cassatt', 'Gustave Caillebotte'
            ]
            
            # アーティスト名から印象派を判定
            df['is_impressionist'] = df['artist'].fillna('').str.contains(
                '|'.join(impressionist_artists), 
                case=False, 
                na=False
            )
            
            # 作品タイトルからも印象派を判定（補完）
            impressionist_keywords = ['impression', 'impressionist', 'monet', 'renoir', 'degas', 'pissarro']
            df.loc[df['title'].fillna('').str.contains('|'.join(impressionist_keywords), case=False, na=False), 'is_impressionist'] = True
            
            # ラベルをエンコード
            y = self.label_encoder.fit_transform(df['is_impressionist'])
            
            self.logger.info(f"データ準備完了 (印象派 vs 非印象派分類):")
            self.logger.info(f"  - 特徴量数: {len(feature_columns)}")
            self.logger.info(f"  - サンプル数: {len(X)}")
            self.logger.info(f"  - クラス分布: {np.bincount(y)}")
        
        return X, y, feature_columns
    
    def train_model(self) -> None:
        """モデルを訓練"""
        self.logger.info("モデル訓練を開始...")
        
        # データを読み込み
        df = self.load_features()
        
        # データを準備
        X, y, feature_names = self.prepare_data(df)
        
        # 訓練・テスト分割
        test_size = self.classification_config['test_size']
        random_state = self.classification_config['random_state']
        
        # インデックスを保持するために分割前にインデックスを取得
        train_indices, test_indices = train_test_split(
            df.index, test_size=test_size, random_state=random_state, stratify=y
        )
        
        X_train = X[train_indices]
        X_test = X[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]
        
        # 訓練データでスケーラーを学習し、同じ変換をテストデータに適用
        self.scaler.fit(X_train)
        X_train = self.scaler.transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        # テストデータのインデックスを保存（誤分類分析用）
        self.test_indices = test_indices
        self.train_indices = train_indices
        
        # ランダムフォレストモデルの作成
        self.model = RandomForestClassifier(
            n_estimators=self.rf_config['n_estimators'],
            max_depth=self.rf_config['max_depth'],
            min_samples_split=self.rf_config['min_samples_split'],
            min_samples_leaf=self.rf_config['min_samples_leaf'],
            random_state=self.rf_config['random_state'],
            n_jobs=-1
        )
        
        # ハイパーパラメータチューニング
        self._tune_hyperparameters(X_train, y_train)
        
        # 最終モデルを訓練
        self.model.fit(X_train, y_train)
        
        # 評価
        self._evaluate_model(df, X_train, X_test, y_train, y_test, feature_names)
        
        # モデルを保存
        self._save_model()
        self._save_scaler()
        
        self.logger.info("モデル訓練完了")
    
    def _tune_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """ハイパーパラメータをチューニング"""
        self.logger.info("ハイパーパラメータチューニングを開始...")
        
        # 各クラスのサンプル数を確認
        class_counts = np.bincount(y_train)
        min_class_count = np.min(class_counts[class_counts > 0])
        
        # サンプル数が少ない場合はハイパーパラメータチューニングをスキップ
        cv_folds = self.classification_config['cv_folds']
        if min_class_count < cv_folds:
            self.logger.warning(f"サンプル数が少ないため、ハイパーパラメータチューニングをスキップします (最小クラス数: {min_class_count}, CV folds: {cv_folds})")
            self.logger.info("デフォルトパラメータでモデルを訓練します")
            return
        
        # グリッドサーチのパラメータ
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # グリッドサーチ実行
        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=cv_folds,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # 最適パラメータを設定
        self.model = grid_search.best_estimator_
        
        self.logger.info(f"最適パラメータ: {grid_search.best_params_}")
        self.logger.info(f"最適スコア: {grid_search.best_score_:.4f}")
    
    def _evaluate_model(self, df: pd.DataFrame, X_train: np.ndarray, X_test: np.ndarray, 
                       y_train: np.ndarray, y_test: np.ndarray, 
                       feature_names: List[str]) -> None:
        """モデルを評価"""
        self.logger.info("モデル評価を開始...")
        
        # 予測
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # 訓練データの評価
        train_accuracy = accuracy_score(y_train, y_train_pred)
        
        # テストデータの評価
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        # 交差検証（サンプル数が十分な場合のみ）
        cv_folds = self.classification_config['cv_folds']
        class_counts = np.bincount(y_train)
        min_class_count = np.min(class_counts[class_counts > 0])
        
        if min_class_count >= cv_folds:
            cv_scores = cross_val_score(self.model, X_train, y_train, 
                                       cv=cv_folds)
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std() * 2
            self.logger.info(f"交差検証スコア: {cv_mean:.4f} (+/- {cv_std:.4f})")
        else:
            cv_scores = np.array([train_accuracy])  # フォールバック
            cv_mean = train_accuracy
            cv_std = 0.0
            self.logger.warning(f"サンプル数が少ないため、交差検証をスキップします (最小クラス数: {min_class_count}, CV folds: {cv_folds})")
        
        # 結果をログに出力
        self.logger.info(f"訓練精度: {train_accuracy:.4f}")
        self.logger.info(f"テスト精度: {test_accuracy:.4f}")
        
        # 詳細な分類レポート（クラス名を決定）
        if self.task_type == 'authenticity':
            # Authentic vs Fake classification
            class_names = ['Authentic (0)', 'Fake (1)']
        else:
            # Impressionist vs Non-Impressionist classification
            class_names = [str(cls) for cls in self.label_encoder.classes_]
        
        report = classification_report(y_test, y_test_pred, target_names=class_names)
        self.logger.info(f"分類レポート:\n{report}")
        
        # 混同行列を可視化
        self._plot_confusion_matrix(y_test, y_test_pred, class_names)
        
        # 特徴量重要度を可視化
        self._plot_feature_importance(feature_names)
        
        # 結果をファイルに保存
        if min_class_count >= cv_folds:
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std() * 2
        else:
            cv_mean = train_accuracy
            cv_std = 0.0
        self._save_results(train_accuracy, test_accuracy, cv_scores, report)
        
        # 誤分類パターンの分析（テストデータのみのデータフレームを作成）
        test_df = df.iloc[self.test_indices].copy()
        self._analyze_misclassifications(test_df, X_test, y_test, y_test_pred)
    
    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              class_names: List[str]) -> None:
        """混同行列を可視化"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('Actual Label')
        
        # 保存
        cm_file = self.viz_dir / 'confusion_matrix.png'
        plt.savefig(cm_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"混同行列を保存: {cm_file}")
    
    def _plot_feature_importance(self, feature_names: List[str]) -> None:
        """特徴量重要度を可視化"""
        importance = self.model.feature_importances_
        
        # 重要度でソート
        indices = np.argsort(importance)[::-1]
        
        # 上位20個の特徴量を表示
        top_n = min(20, len(feature_names))
        
        plt.figure(figsize=(10, 8))
        plt.title(f'Feature Importance (Top {top_n})')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.bar(range(top_n), importance[indices[:top_n]])
        plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], 
                  rotation=45, ha='right')
        plt.tight_layout()
        
        # 保存
        importance_file = self.viz_dir / 'feature_importance.png'
        plt.savefig(importance_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"特徴量重要度を保存: {importance_file}")
    
    def _save_results(self, train_accuracy: float, test_accuracy: float, 
                     cv_scores: np.ndarray, report: str) -> None:
        """結果をファイルに保存"""
        with open(self.results_file, 'w', encoding='utf-8') as f:
            f.write("ランダムフォレスト訓練結果\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"訓練精度: {train_accuracy:.4f}\n")
            f.write(f"テスト精度: {test_accuracy:.4f}\n")
            f.write(f"交差検証スコア: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})\n\n")
            f.write("分類レポート:\n")
            f.write(report)
    
    def _save_model(self) -> None:
        """モデルを保存"""
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns
        }
        
        with open(self.model_file, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"モデルを保存: {self.model_file}")
    
    def _save_scaler(self) -> None:
        """訓練データでfitしたスケーラーを保存"""
        scaler_file = self.timestamp_manager.get_scaler_file()
        with open(scaler_file, 'wb') as f:
            pickle.dump(self.scaler, f)
        self.logger.info(f"スケーラーを保存: {scaler_file}")
    
    def load_model(self) -> None:
        """保存されたモデルを読み込み"""
        if not self.model_file.exists():
            raise FileNotFoundError(f"モデルファイルが見つかりません: {self.model_file}")
        
        with open(self.model_file, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.feature_columns = model_data['feature_columns']
        
        self.logger.info("モデル読み込み完了")
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        予測を実行
        
        Args:
            X: 特徴量配列
            
        Returns:
            predictions: 予測ラベル
            probabilities: 予測確率
        """
        if self.model is None:
            raise ValueError("モデルが訓練されていません")
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        return predictions, probabilities
    
    def _analyze_misclassifications(self, df: pd.DataFrame, X_test: np.ndarray, 
                                    y_test: np.ndarray, y_test_pred: np.ndarray) -> None:
        """
        誤分類パターンを分析
        
        Args:
            df: データフレーム（全データ）
            X_test: テストデータの特徴量
            y_test: テストデータの正解ラベル
            y_test_pred: テストデータの予測ラベル
        """
        self.logger.info("誤分類パターンを分析中...")
        
        # 誤分類を特定
        misclassified_mask = y_test != y_test_pred
        misclassified_indices = np.where(misclassified_mask)[0]
        
        if len(misclassified_indices) == 0:
            self.logger.info("誤分類はありませんでした")
            return
        
        self.logger.info(f"誤分類数: {len(misclassified_indices)} / {len(y_test)} ({len(misclassified_indices)/len(y_test)*100:.2f}%)")
        
        # 誤分類タイプを分類
        if self.task_type == 'authenticity':
            # 本物 vs フェイク分類の場合
            # 偽陽性: 本物(0)をフェイク(1)と誤分類
            # 偽陰性: フェイク(1)を本物(0)と誤分類
            false_positives = np.where((y_test == 0) & (y_test_pred == 1))[0]
            false_negatives = np.where((y_test == 1) & (y_test_pred == 0))[0]
            
            self.logger.info(f"偽陽性（本物をフェイクと誤分類）: {len(false_positives)}")
            self.logger.info(f"偽陰性（フェイクを本物と誤分類）: {len(false_negatives)}")
            
            # 予測確率を取得
            y_test_proba = self.model.predict_proba(X_test)
            
            # 偽陽性の分析（本物がフェイクと誤分類された）
            if len(false_positives) > 0:
                fp_proba = y_test_proba[false_positives, 1]  # フェイクとして予測された確率
                self.logger.info(f"偽陽性の平均確率: {np.mean(fp_proba):.3f} (最小: {np.min(fp_proba):.3f}, 最大: {np.max(fp_proba):.3f})")
            
            # 偽陰性の分析（フェイクが本物と誤分類された）
            if len(false_negatives) > 0:
                fn_proba = y_test_proba[false_negatives, 0]  # 本物として予測された確率
                self.logger.info(f"偽陰性の平均確率: {np.mean(fn_proba):.3f} (最小: {np.min(fn_proba):.3f}, 最大: {np.max(fn_proba):.3f})")
            
            # 誤分類結果をファイルに保存
            misclassified_results = []
            
            for idx in misclassified_indices:
                actual_label = 'Authentic' if y_test[idx] == 0 else 'Fake'
                predicted_label = 'Authentic' if y_test_pred[idx] == 0 else 'Fake'
                prob = y_test_proba[idx, 1] if y_test_pred[idx] == 1 else y_test_proba[idx, 0]
                
                result = {
                    'test_index': idx,
                    'actual_label': actual_label,
                    'predicted_label': predicted_label,
                    'predicted_probability': prob
                }
                
                # データフレームから情報を追加（test_dfのインデックスは0から始まる）
                if idx < len(df):
                    row = df.iloc[idx]
                    if 'image_id' in df.columns:
                        result['image_id'] = row.get('image_id', 'unknown')
                    if 'source' in df.columns:
                        result['source'] = row.get('source', 'unknown')
                    if 'generation_model' in df.columns:
                        result['generation_model'] = row.get('generation_model', 'unknown')
                
                misclassified_results.append(result)
            
            misclassified_df = pd.DataFrame(misclassified_results)
            misclassified_file = self.results_dir / 'misclassifications.csv'
            misclassified_df.to_csv(misclassified_file, index=False)
            
            self.logger.info(f"誤分類結果を保存: {misclassified_file}")
        
        else:
            # 印象派 vs 非印象派分類の場合
            false_positives = np.where((y_test == 0) & (y_test_pred == 1))[0]
            false_negatives = np.where((y_test == 1) & (y_test_pred == 0))[0]
            
            self.logger.info(f"偽陽性（非印象派を印象派と誤分類）: {len(false_positives)}")
            self.logger.info(f"偽陰性（印象派を非印象派と誤分類）: {len(false_negatives)}")
        
        # 誤分類の可視化
        self._plot_misclassification_analysis(y_test, y_test_pred, false_positives, false_negatives)
    
    def _plot_misclassification_analysis(self, y_test: np.ndarray, y_test_pred: np.ndarray,
                                         false_positives: np.ndarray, false_negatives: np.ndarray) -> None:
        """誤分類分析の可視化"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        if self.task_type == 'authenticity':
            labels = ['Authentic (0)', 'Fake (1)']
            fp_label = 'Authentic→Fake'
            fn_label = 'Fake→Authentic'
        else:
            labels = ['Non-Impressionist', 'Impressionist']
            fp_label = 'Non-Impressionist→Impressionist'
            fn_label = 'Impressionist→Non-Impressionist'
        
        # 誤分類数の比較
        ax1 = axes[0]
        categories = [fp_label, fn_label]
        counts = [len(false_positives), len(false_negatives)]
        colors = ['#ff6b6b', '#4ecdc4']
        
        bars = ax1.bar(categories, counts, color=colors)
        ax1.set_ylabel('Misclassification Count', fontsize=12)
        ax1.set_title('Misclassification by Type', fontsize=14)
        ax1.grid(axis='y', alpha=0.3)
        
        # バーの上に値を表示
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}',
                    ha='center', va='bottom', fontsize=11)
        
        # 混同行列の拡大（誤分類部分）
        ax2 = axes[1]
        
        # 簡易的な混同行列
        cm_data = [[len(y_test[(y_test == 0) & (y_test_pred == 0)]), len(false_positives)],
                  [len(false_negatives), len(y_test[(y_test == 1) & (y_test_pred == 1)])]]
        
        sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels,
                   ax=ax2, cbar_kws={'label': 'Sample Count'})
        ax2.set_title('Confusion Matrix (Misclassifications Highlighted)', fontsize=14)
        ax2.set_xlabel('Predicted Label', fontsize=12)
        ax2.set_ylabel('Actual Label', fontsize=12)
        
        plt.tight_layout()
        
        # 保存
        misclassification_file = self.viz_dir / 'misclassification_analysis.png'
        plt.savefig(misclassification_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"誤分類分析を保存: {misclassification_file}")
