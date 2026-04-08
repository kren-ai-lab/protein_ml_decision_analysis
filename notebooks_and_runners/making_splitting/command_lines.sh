python embedding_analysis_space.py \
  --emb-train ../../processed_dataset/encoded_data/esm2_t6_8M_UR50D/full_data.csv \
  --output-dir ../../processed_dataset/redundancy_analysis/ \
  --label-col label \
  --sequence-col sequence \
  --id-col id \
  --feature-prefix p_ \
  --prefix esm2_t6


python run_biosieve_reducers_from_percentiles.py \
  --embedding-npy ../../processed_dataset/redundancy_analysis/artifacts/training_embeddings.npy \
  --percentiles-csv ../../processed_dataset/redundancy_analysis/tables/esm2_t6_similarity_percentiles.csv \
  --input-data ../../processed_dataset/encoded_data/esm2_t6_8M_UR50D/full_data.csv \
  --output-dir ../../processed_dataset/reduction_process/ \
  --label-col label \
  --id-col id

python run_biosieve_splits_from_reduction_summary.py \
  --reduction-summary ../../processed_dataset/reduction_process/reduction_summary.csv \
  --reduction-dir ../../processed_dataset/reduction_process \
  --full-data ../../processed_dataset/encoded_data/esm2_t6_8M_UR50D/full_data.csv \
  --output-dir ../../processed_dataset/split_process \
  --label-col label \
  --id-col id \
  --feature-prefix p_ \
  --n-splits 5 \
  --seed 13 \
  --val-size 0.1 \
  --metric cosine \
  --shuffle-ties
