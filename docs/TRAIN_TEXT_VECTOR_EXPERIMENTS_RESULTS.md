# train_text_vector_experiments 途中結果

時系列 CV（val years = 2013〜2016）でのテキストベクトル化比較。**nmf_20 の途中で停止したため、ここまでを記録し再開用に残す。**

## 記録済み結果（〜 nmf_10 まで完了）

| config       | n_feat | CV_AUC  |
|-------------|--------|---------|
| base        | 34     | 0.7600  |
| tfidf_50    | 84     | 0.7600  |
| tfidf_100   | 134    | 0.7600  |
| tfidf_svd20 | 54     | 0.7601  |
| count_50    | 84     | 0.7603  |
| count_svd20 | 54     | 0.7603  |
| hash_50     | 84     | 0.7600  |
| hash_svd20  | 54     | 0.7601  |
| lda_10      | 44     | 0.7606  |
| lda_20      | 54     | 0.7603  |
| nmf_10      | 44     | 0.7600  |

- **nmf_20** は実行途中で停止（未記録）。
- 続きは **nmf_20 から** 実行すればよい。

## 再開方法

ノート内の「再開用（任意）」セルで以下を設定した状態で、その下の CV セルを実行する。

- `RECORDED_PARTIAL_RESULTS`: 上記の記録済み結果のリスト（セルに記載済み）。
- `START_FROM_CONFIG = "nmf_20"`: ここから実行。全件やり直す場合は `RECORDED_PARTIAL_RESULTS = []` と `START_FROM_CONFIG = None` にする。
