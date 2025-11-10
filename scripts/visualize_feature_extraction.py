#!/usr/bin/env python3
"""
特徴量抽出の可視化スクリプト
実際の画像を使って、各特徴量がどのように抽出されるかを可視化する
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from src.utils.config_manager import ConfigManager
from src.feature_extraction.color_extractor import ColorFeatureExtractor


def visualize_color_features(original_path: Path, generated_path: Path, output_path: Path):
    """色彩特徴量の可視化"""
    # 画像を読み込み
    original = cv2.imread(str(original_path))
    generated = cv2.imread(str(generated_path))
    
    if original is None or generated is None:
        raise FileNotFoundError(f"画像が見つかりません: {original_path} or {generated_path}")
    
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    generated = cv2.cvtColor(generated, cv2.COLOR_BGR2RGB)
    
    # リサイズ
    original = cv2.resize(original, (512, 512))
    generated = cv2.resize(generated, (512, 512))
    
    # HSV変換
    original_hsv = cv2.cvtColor(original, cv2.COLOR_RGB2HSV)
    generated_hsv = cv2.cvtColor(generated, cv2.COLOR_RGB2HSV)
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # 元画像
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(original)
    ax1.set_title('Original Image', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(generated)
    ax2.set_title('Generated Image', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # HSV色空間の可視化
    for i, (name, idx) in enumerate([('Hue (色相)', 0), ('Saturation (彩度)', 1), ('Value (明度)', 2)]):
        ax_orig = fig.add_subplot(gs[1, i])
        ax_gen = fig.add_subplot(gs[2, i])
        
        orig_channel = original_hsv[:, :, idx]
        gen_channel = generated_hsv[:, :, idx]
        
        ax_orig.imshow(orig_channel, cmap='viridis' if idx == 0 else 'gray')
        ax_orig.set_title(f'Original {name}\nMean: {np.mean(orig_channel):.1f}, Std: {np.std(orig_channel):.1f}', 
                         fontsize=10)
        ax_orig.axis('off')
        
        ax_gen.imshow(gen_channel, cmap='viridis' if idx == 0 else 'gray')
        ax_gen.set_title(f'Generated {name}\nMean: {np.mean(gen_channel):.1f}, Std: {np.std(gen_channel):.1f}', 
                        fontsize=10)
        ax_gen.axis('off')
    
    # 色相ヒストグラム
    ax_hist = fig.add_subplot(gs[1, 3])
    orig_hue = original_hsv[:, :, 0].ravel()
    gen_hue = generated_hsv[:, :, 0].ravel()
    ax_hist.hist(orig_hue, bins=50, alpha=0.7, label='Original', color='blue')
    ax_hist.hist(gen_hue, bins=50, alpha=0.7, label='Generated', color='red')
    ax_hist.set_xlabel('Hue Value')
    ax_hist.set_ylabel('Frequency')
    ax_hist.set_title('Hue Distribution')
    ax_hist.legend()
    
    # 彩度ヒストグラム
    ax_hist2 = fig.add_subplot(gs[2, 3])
    orig_sat = original_hsv[:, :, 1].ravel()
    gen_sat = generated_hsv[:, :, 1].ravel()
    ax_hist2.hist(orig_sat, bins=50, alpha=0.7, label='Original', color='blue')
    ax_hist2.hist(gen_sat, bins=50, alpha=0.7, label='Generated', color='red')
    ax_hist2.set_xlabel('Saturation Value')
    ax_hist2.set_ylabel('Frequency')
    ax_hist2.set_title('Saturation Distribution')
    ax_hist2.legend()
    
    plt.suptitle('Color Feature Extraction: HSV Color Space Analysis', fontsize=14, fontweight='bold', y=0.98)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_edge_features(original_path: Path, generated_path: Path, output_path: Path):
    """エッジ特徴量の可視化"""
    # 画像を読み込み
    original = cv2.imread(str(original_path))
    generated = cv2.imread(str(generated_path))
    
    if original is None or generated is None:
        raise FileNotFoundError(f"画像が見つかりません: {original_path} or {generated_path}")
    
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    generated = cv2.cvtColor(generated, cv2.COLOR_BGR2RGB)
    
    # リサイズ
    original = cv2.resize(original, (512, 512))
    generated = cv2.resize(generated, (512, 512))
    
    # グレースケール変換
    original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    generated_gray = cv2.cvtColor(generated, cv2.COLOR_RGB2GRAY)
    
    # Cannyエッジ検出
    original_edges = cv2.Canny(original_gray, 50, 150)
    generated_edges = cv2.Canny(generated_gray, 50, 150)
    
    # エッジコンポーネントの分析
    orig_num_labels, orig_labels, orig_stats, _ = cv2.connectedComponentsWithStats(original_edges, connectivity=8)
    gen_num_labels, gen_labels, gen_stats, _ = cv2.connectedComponentsWithStats(generated_edges, connectivity=8)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 元画像
    axes[0, 0].imshow(original)
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(generated)
    axes[1, 0].set_title('Generated Image', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # エッジ検出結果
    axes[0, 1].imshow(original_edges, cmap='gray')
    axes[0, 1].set_title(f'Original Edges\nCount: {orig_num_labels-1} components', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[1, 1].imshow(generated_edges, cmap='gray')
    axes[1, 1].set_title(f'Generated Edges\nCount: {gen_num_labels-1} components', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    # エッジを重ね合わせ
    orig_overlay = original.copy()
    orig_overlay[original_edges > 0] = [255, 0, 0]  # 赤でエッジを表示
    axes[0, 2].imshow(orig_overlay)
    axes[0, 2].set_title('Original with Edges Overlay', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    gen_overlay = generated.copy()
    gen_overlay[generated_edges > 0] = [255, 0, 0]  # 赤でエッジを表示
    axes[1, 2].imshow(gen_overlay)
    axes[1, 2].set_title('Generated with Edges Overlay', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.suptitle('Edge Feature Extraction: Canny Edge Detection', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_texture_features(original_path: Path, generated_path: Path, output_path: Path):
    """テクスチャ特徴量の可視化"""
    from skimage.feature import local_binary_pattern
    
    # 画像を読み込み
    original = cv2.imread(str(original_path))
    generated = cv2.imread(str(generated_path))
    
    if original is None or generated is None:
        raise FileNotFoundError(f"画像が見つかりません: {original_path} or {generated_path}")
    
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    generated = cv2.cvtColor(generated, cv2.COLOR_BGR2RGB)
    
    # リサイズ
    original = cv2.resize(original, (512, 512))
    generated = cv2.resize(generated, (512, 512))
    
    # グレースケール変換
    original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    generated_gray = cv2.cvtColor(generated, cv2.COLOR_RGB2GRAY)
    
    # LBP計算
    radius = 1
    n_points = 8 * radius
    original_lbp = local_binary_pattern(original_gray, n_points, radius, method='uniform')
    generated_lbp = local_binary_pattern(generated_gray, n_points, radius, method='uniform')
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 元画像
    axes[0, 0].imshow(original)
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(generated)
    axes[1, 0].set_title('Generated Image', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # LBP可視化
    axes[0, 1].imshow(original_lbp, cmap='gray')
    axes[0, 1].set_title(f'Original LBP\nMean: {np.mean(original_lbp):.2f}, Std: {np.std(original_lbp):.2f}', 
                        fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[1, 1].imshow(generated_lbp, cmap='gray')
    axes[1, 1].set_title(f'Generated LBP\nMean: {np.mean(generated_lbp):.2f}, Std: {np.std(generated_lbp):.2f}', 
                         fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    # LBPヒストグラム
    axes[0, 2].hist(original_lbp.ravel(), bins=50, alpha=0.7, color='blue')
    axes[0, 2].set_xlabel('LBP Value')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Original LBP Histogram')
    
    axes[1, 2].hist(generated_lbp.ravel(), bins=50, alpha=0.7, color='red')
    axes[1, 2].set_xlabel('LBP Value')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('Generated LBP Histogram')
    
    plt.suptitle('Texture Feature Extraction: Local Binary Pattern (LBP)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="特徴量抽出の可視化")
    parser.add_argument("--image-id", type=str, default="9733",
                        help="使用する画像ID（デフォルト: 9733）")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="出力ディレクトリ（デフォルト: data/artifacts/visualizations/latest）")
    args = parser.parse_args()
    
    config = ConfigManager().get_config()
    
    # パス設定
    base_path = Path(config.get('wikiart_vlm', {}).get('base_path', 'data/external/WikiArt_VLM-main'))
    original_path = base_path / 'images' / 'Original' / f"{args.image_id}.jpg"
    generated_path = base_path / 'images' / 'Stable-Diffusion' / f"{args.image_id}.jpg"
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        data_config = config['data']
        artifacts_root = Path(data_config['output_dir'])
        artifacts_subdir = data_config.get('artifacts_subdir', '').strip()
        if artifacts_subdir:
            artifacts_root = artifacts_root / artifacts_subdir
        output_dir = artifacts_root / 'visualizations' / data_config.get('latest_label', 'latest')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not original_path.exists() or not generated_path.exists():
        print(f"エラー: 画像が見つかりません")
        print(f"  Original: {original_path}")
        print(f"  Generated: {generated_path}")
        return
    
    print(f"画像を読み込み: {args.image_id}")
    print(f"  Original: {original_path}")
    print(f"  Generated: {generated_path}")
    
    # 可視化を生成
    try:
        print("\n色彩特徴量の可視化を生成中...")
        color_output = output_dir / f"feature_visualization_color_{args.image_id}.png"
        visualize_color_features(original_path, generated_path, color_output)
        print(f"  保存: {color_output}")
        
        print("\nエッジ特徴量の可視化を生成中...")
        edge_output = output_dir / f"feature_visualization_edge_{args.image_id}.png"
        visualize_edge_features(original_path, generated_path, edge_output)
        print(f"  保存: {edge_output}")
        
        print("\nテクスチャ特徴量の可視化を生成中...")
        texture_output = output_dir / f"feature_visualization_texture_{args.image_id}.png"
        visualize_texture_features(original_path, generated_path, texture_output)
        print(f"  保存: {texture_output}")
        
        print("\n可視化完了!")
        
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
