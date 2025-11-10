# docs/ ディレクトリガイド

`docs/` 配下のファイルは役割ごとに4つのサブディレクトリへ整理されています。必要な情報を迷わず参照できるよう、カテゴリ別に用途をまとめています。

## ディレクトリ一覧

| ディレクトリ | 目的 | 代表的なファイル |
| --- | --- | --- |
| `history/` | 開発履歴・コミットログ | `cursor_chat_history.md`, `git_commit_status.md` |
| `reports/` | 実装・演習レポートやデータリーク調査 | `implementation_status_report.md`, `exercise_report.html`, `data_leakage_*.md` |
| `guides/` | 手順書・技術メモ | `feature_columns_definition.md`, `feature_extraction_cheatsheet.md`, `timestamp_management.md`, `loop_prevention_refactoring.md` |
| `plans/` | 将来計画・提案資料 | `phase4_implementation.md`, `vlm_model_recommendations.md`, `生成AIによるフェイクアート判別と芸術倫理教育プロジェクト企画書*.md` |

## 運用ポリシー

- レポートや検証結果は「reports」、開発ナレッジやベストプラクティスは「guides」に配置します。
- 時系列で追いたい情報（チャット履歴・コミット状況）は「history」に集約しています。
- 新しい計画・提案資料は「plans」に追加し、完了後もアーカイブとして残します。

この構成により、「どの文書をどこに置くか」をメンバー全員で共有しやすくなりました。
