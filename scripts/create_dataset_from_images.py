# -*- coding: utf-8 -*-
"""
画像ファイル名をもとにデータセットを作成するスクリプト
paintings_imagesにある画像ファイルのObject_IDを使って、raw_dataから対応するデータを抽出
"""

import sys
from pathlib import Path
import pandas as pd

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config_manager import ConfigManager
from src.utils.logger import setup_logging, get_logger


def extract_object_ids_from_images(images_dir):
    """画像ファイル名からObject_IDを抽出"""
    images_dir = Path(images_dir)
    
    # 画像ファイルを取得（.jpg, .JPG両方に対応）
    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.JPG'))
    
    # Object_IDを抽出（ファイル名から拡張子を除いた部分）
    object_ids = []
    for img_file in image_files:
        object_id_str = img_file.stem  # 拡張子を除いたファイル名
        try:
            object_id = int(object_id_str)
            object_ids.append(object_id)
        except ValueError:
            # 数値に変換できない場合はスキップ
            continue
    
    return sorted(set(object_ids))  # 重複を削除してソート


def create_dataset_from_raw_data(object_ids, raw_csv_path, output_path):
    """raw_dataのCSVからObject_IDに一致するデータを抽出してデータセットを作成"""
    logger = get_logger(__name__)
    
    logger.info(f"raw_data CSVを読み込み中: {raw_csv_path}")
    logger.info(f"対象Object_ID数: {len(object_ids):,}件")
    
    # Object_IDをセットに変換（高速検索のため）
    object_ids_set = set(object_ids)
    
    # チャンクごとに読み込んで処理（大容量ファイル対応）
    chunks = []
    chunk_size = 10000
    matched_count = 0
    
    logger.info("CSVファイルを処理中...")
    for chunk_idx, chunk in enumerate(pd.read_csv(raw_csv_path, chunksize=chunk_size, low_memory=False)):
        # Object_ID列でフィルタリング
        # Object_ID列が存在するか確認
        if 'Object ID' in chunk.columns:
            object_id_col = 'Object ID'
        elif 'Object_ID' in chunk.columns:
            object_id_col = 'Object_ID'
        else:
            # 列名を探す
            id_cols = [col for col in chunk.columns if 'object' in col.lower() and 'id' in col.lower()]
            if id_cols:
                object_id_col = id_cols[0]
                logger.info(f"Object_ID列を検出: {object_id_col}")
            else:
                logger.error("Object_ID列が見つかりません")
                return None
        
        # Object_IDでフィルタリング
        filtered_chunk = chunk[chunk[object_id_col].isin(object_ids_set)]
        
        if len(filtered_chunk) > 0:
            chunks.append(filtered_chunk)
            matched_count += len(filtered_chunk)
            logger.info(f"チャンク {chunk_idx + 1}: {len(filtered_chunk)}件のデータを抽出（累計: {matched_count:,}件）")
        
        # 全て見つかったら早期終了
        if matched_count >= len(object_ids):
            logger.info("全てのObject_IDが見つかりました")
            break
    
    if not chunks:
        logger.warning("一致するデータが見つかりませんでした")
        return None
    
    # チャンクを結合
    logger.info("データを結合中...")
    df = pd.concat(chunks, ignore_index=True)
    
    # 重複を削除（念のため）
    df = df.drop_duplicates(subset=[object_id_col], keep='first')
    
    logger.info(f"データセット作成完了: {len(df):,}行 × {len(df.columns)}列")
    
    # データセットを保存
    logger.info(f"データセットを保存中: {output_path}")
    df.to_csv(output_path, index=False, encoding='utf-8')
    logger.info(f"データセット保存完了: {output_path}")
    
    # 統計情報
    logger.info("=" * 60)
    logger.info("データセット作成サマリー")
    logger.info("=" * 60)
    logger.info(f"画像ファイル数: {len(object_ids):,}件")
    logger.info(f"抽出されたデータ: {len(df):,}件")
    logger.info(f"マッチング率: {(len(df) / len(object_ids) * 100):.1f}%")
    
    return df


def main():
    """メイン関数"""
    # 設定読み込み
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    # ログ設定
    setup_logging(
        log_file='create_dataset_from_images.log',
        config=config
    )
    logger = get_logger(__name__)
    
    logger.info("=" * 60)
    logger.info("画像ファイル名からデータセット作成を開始")
    logger.info("=" * 60)
    
    # パス設定
    images_dir = Path("data/filtered_data/paintings_images")
    raw_csv_path = Path("data/raw_data/MetObjects.csv")
    output_path = Path("data/filtered_data/dataset_from_images.csv")
    
    # 画像ファイルからObject_IDを抽出
    logger.info("画像ファイル名からObject_IDを抽出中...")
    object_ids = extract_object_ids_from_images(images_dir)
    logger.info(f"抽出されたObject_ID: {len(object_ids):,}件")
    
    if not object_ids:
        logger.error("画像ファイルからObject_IDを抽出できませんでした")
        return False
    
    # サンプルを表示
    logger.info(f"Object_IDサンプル: {object_ids[:10]}")
    
    # raw_dataからデータセットを作成
    df = create_dataset_from_raw_data(object_ids, raw_csv_path, output_path)
    
    if df is None:
        logger.error("データセットの作成に失敗しました")
        return False
    
    logger.info("=" * 60)
    logger.info("データセット作成完了")
    logger.info("=" * 60)
    logger.info(f"出力ファイル: {output_path}")
    
    print(f"\nデータセット作成が完了しました。")
    print(f"出力ファイル: {output_path}")
    print(f"データ行数: {len(df):,}行")
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)

