#!/usr/bin/env python3
"""
WikiArt_VLMデータセットのバイアス可視化スクリプト

ラベル（0: Authentic, 1: Fake）ごとの主要特徴量統計と分布図を作成し、
学習者に「なぜ簡単に分類できるのか」を示す補助教材として利用する。
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils.config_manager import ConfigManager
from src.utils.timestamp_manager import TimestampManager


def detect_features_file(features_dir: Path,
                         generation_model: str,
                         include_gestalt: bool = True) -> Path:
    """ディレクトリ内から特徴量CSVを自動検出"""
    suffix = "_with_gestalt" if include_gestalt else ""
    pattern = f"wikiart_vlm_features{suffix}_{generation_model}*.csv"
    candidates = sorted(features_dir.glob(pattern))
    if not candidates and include_gestalt:
        # フォールバック：ゲシュタルト無し
        return detect_features_file(features_dir, generation_model, include_gestalt=False)
    if not candidates:
        raise FileNotFoundError(
            f"特徴量ファイルが見つかりません: {features_dir} / {pattern}"
        )
    return candidates[0]


def compute_feature_summary(df: pd.DataFrame,
                            exclude: Optional[List[str]] = None) -> pd.DataFrame:
    """ラベル別に平均・標準偏差をまとめたDataFrameを作成"""
    exclude = set(exclude or [])
    numeric_cols = [
        col for col in df.select_dtypes(include=["number"]).columns
        if col not in exclude
    ]
    summary_rows = []
    authentic = df[df["label"] == 0]
    fake = df[df["label"] == 1]

    for col in numeric_cols:
        mean_a = authentic[col].mean()
        mean_f = fake[col].mean()
        std_a = authentic[col].std()
        std_f = fake[col].std()
        summary_rows.append({
            "feature": col,
            "mean_authentic": mean_a,
            "mean_fake": mean_f,
            "std_authentic": std_a,
            "std_fake": std_f,
            "mean_diff": mean_f - mean_a,
            "mean_diff_abs": abs(mean_f - mean_a),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.sort_values("mean_diff_abs", ascending=False, inplace=True)
    return summary_df


def plot_distributions(df: pd.DataFrame,
                       features: List[str],
                       output_path: Path) -> None:
    """特徴量の分布をラベル別に描画"""
    sns.set_style("whitegrid")
    label_map = {0: "Authentic (0)", 1: "Fake (1)"}
    df_plot = df.copy()
    df_plot["Label"] = df_plot["label"].map(label_map)

    n_features = len(features)
    n_cols = 2
    n_rows = (n_features + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes = axes.flatten()

    for idx, feature in enumerate(features):
        ax = axes[idx]
        sns.kdeplot(
            data=df_plot,
            x=feature,
            hue="Label",
            fill=True,
            common_norm=False,
            alpha=0.4,
            ax=ax
        )
        ax.set_title(feature)
        ax.set_ylabel("Density")

    # 残りのサブプロットを非表示
    for j in range(idx + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Top Feature Distributions by Label", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_mean_diffs(summary_df: pd.DataFrame,
                    features: List[str],
                    output_path: Path) -> None:
    """平均値の差を棒グラフで表示"""
    subset = summary_df.set_index("feature").loc[features].reset_index()
    subset["mean_diff_pct"] = subset["mean_diff"]

    plt.figure(figsize=(10, 4 + len(features) * 0.4))
    sns.barplot(
        data=subset,
        y="feature",
        x="mean_diff",
        palette="RdBu",
        hue=(subset["mean_diff"] > 0).map({True: "Fake > Authentic", False: "Authentic > Fake"})
    )
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Mean Difference (Fake - Authentic)")
    plt.ylabel("Feature")
    plt.title("Mean Difference per Feature")
    plt.legend(title="", loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="WikiArt_VLMデータのバイアスを可視化し、主要特徴量の統計を出力します。"
    )
    parser.add_argument("--features-file", type=str, default=None,
                        help="特徴量CSVのパス（未指定時はlatestディレクトリから自動検出）")
    parser.add_argument("--generation-model", type=str, default="Stable-Diffusion",
                        help="使用する生成モデル名（ファイル自動検出に利用）")
    parser.add_argument("--top-k", type=int, default=6,
                        help="分布図および棒グラフに表示する特徴量数")
    parser.add_argument("--use-shap-importance", action="store_true",
                        help="SHAP特徴量重要度ファイルを使用して特徴量を選択（未指定時は平均値の差の絶対値で選択）")
    parser.add_argument("--shap-importance-file", type=str, default=None,
                        help="SHAP特徴量重要度CSVのパス（未指定時はlatestディレクトリから自動検出）")
    args = parser.parse_args()

    config = ConfigManager().get_config()
    timestamp_manager = TimestampManager(config)
    timestamp_manager.create_directories()

    if args.features_file:
        features_path = Path(args.features_file)
    else:
        features_dir = timestamp_manager.get_features_dir()
        features_path = detect_features_file(features_dir, args.generation_model)

    if not features_path.exists():
        raise FileNotFoundError(f"特徴量ファイルが見つかりません: {features_path}")

    df = pd.read_csv(features_path)
    if "label" not in df.columns:
        raise ValueError("特徴量ファイルに'label'カラムが必要です。")

    summary_df = compute_feature_summary(df, exclude=["image_id"])
    
    # SHAP特徴量重要度を使用する場合
    if args.use_shap_importance:
        if args.shap_importance_file:
            shap_path = Path(args.shap_importance_file)
        else:
            shap_dir = timestamp_manager.get_shap_explanations_dir()
            shap_path = shap_dir / "feature_importance_shap.csv"
        
        if not shap_path.exists():
            raise FileNotFoundError(f"SHAP特徴量重要度ファイルが見つかりません: {shap_path}")
        
        shap_df = pd.read_csv(shap_path)
        top_features = shap_df.head(args.top_k)["feature"].tolist()
        print(f"Using SHAP importance to select top {args.top_k} features")
        print(f"Selected features: {top_features}")
    else:
        top_features = summary_df.head(args.top_k)["feature"].tolist()
        print(f"Using mean difference to select top {args.top_k} features")
        print(f"Selected features: {top_features}")

    results_dir = timestamp_manager.get_results_dir()
    viz_dir = timestamp_manager.get_visualizations_dir()

    summary_path = results_dir / "bias_feature_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    distributions_path = viz_dir / "bias_feature_distributions.png"
    plot_distributions(df, top_features, distributions_path)

    mean_diff_path = viz_dir / "bias_feature_mean_diff.png"
    plot_mean_diffs(summary_df, top_features, mean_diff_path)

    print("=== Bias Summary ===")
    print(summary_df.head(args.top_k).to_string(index=False))
    print(f"\nSaved summary CSV: {summary_path}")
    print(f"Saved distribution plot: {distributions_path}")
    print(f"Saved mean diff plot: {mean_diff_path}")


if __name__ == "__main__":
    main()
