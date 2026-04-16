python run_biosieve_splits_from_reduction_summary.py \
  --reduction-summary ../../reduced_dataset/not_normalized_data/ankh2-ext1/reduction_summary.csv \
  --reduction-dir ../../reduced_dataset/not_normalized_data/ankh2-ext1/ \
  --full-data ../../reduced_dataset/not_normalized_data/ankh2-ext1/full_data_with_biosieve_id.csv \
  --output-dir ../../split_process/not_normalized_data/ankh2-ext1 \
  --label-col label \
  --id-col id \
  --feature-prefix p_ \
  --n-splits 5 \
  --seed 13 \
  --val-size 0.1 \
  --metric cosine \
  --shuffle-ties


python run_biosieve_splits_from_reduction_summary.py \
  --reduction-summary ../../reduced_dataset/not_normalized_data/ankh3-large/reduction_summary.csv \
  --reduction-dir ../../reduced_dataset/not_normalized_data/ankh3-large/ \
  --full-data ../../reduced_dataset/not_normalized_data/ankh3-large/full_data_with_biosieve_id.csv \
  --output-dir ../../split_process/not_normalized_data/ankh3-large \
  --label-col label \
  --id-col id \
  --feature-prefix p_ \
  --n-splits 5 \
  --seed 13 \
  --val-size 0.1 \
  --metric cosine \
  --shuffle-ties

python run_biosieve_splits_from_reduction_summary.py \
  --reduction-summary ../../reduced_dataset/not_normalized_data/esm2_t6_8M_UR50D/reduction_summary.csv \
  --reduction-dir ../../reduced_dataset/not_normalized_data/esm2_t6_8M_UR50D/ \
  --full-data ../../reduced_dataset/not_normalized_data/esm2_t6_8M_UR50D/full_data_with_biosieve_id.csv \
  --output-dir ../../split_process/not_normalized_data/esm2_t6_8M_UR50D \
  --label-col label \
  --id-col id \
  --feature-prefix p_ \
  --n-splits 5 \
  --seed 13 \
  --val-size 0.1 \
  --metric cosine \
  --shuffle-ties

python run_biosieve_splits_from_reduction_summary.py \
  --reduction-summary ../../reduced_dataset/not_normalized_data/esm2_t12_35M_UR50D/reduction_summary.csv \
  --reduction-dir ../../reduced_dataset/not_normalized_data/esm2_t12_35M_UR50D/ \
  --full-data ../../reduced_dataset/not_normalized_data/esm2_t12_35M_UR50D/full_data_with_biosieve_id.csv \
  --output-dir ../../split_process/not_normalized_data/esm2_t12_35M_UR50D \
  --label-col label \
  --id-col id \
  --feature-prefix p_ \
  --n-splits 5 \
  --seed 13 \
  --val-size 0.1 \
  --metric cosine \
  --shuffle-ties

python run_biosieve_splits_from_reduction_summary.py \
  --reduction-summary ../../reduced_dataset/not_normalized_data/esm2_t30_150M_UR50D/reduction_summary.csv \
  --reduction-dir ../../reduced_dataset/not_normalized_data/esm2_t30_150M_UR50D/ \
  --full-data ../../reduced_dataset/not_normalized_data/esm2_t30_150M_UR50D/full_data_with_biosieve_id.csv \
  --output-dir ../../split_process/not_normalized_data/esm2_t30_150M_UR50D \
  --label-col label \
  --id-col id \
  --feature-prefix p_ \
  --n-splits 5 \
  --seed 13 \
  --val-size 0.1 \
  --metric cosine \
  --shuffle-ties

python run_biosieve_splits_from_reduction_summary.py \
  --reduction-summary ../../reduced_dataset/not_normalized_data/esm2_t33_650M_UR50D/reduction_summary.csv \
  --reduction-dir ../../reduced_dataset/not_normalized_data/esm2_t33_650M_UR50D/ \
  --full-data ../../reduced_dataset/not_normalized_data/esm2_t33_650M_UR50D/full_data_with_biosieve_id.csv \
  --output-dir ../../split_process/not_normalized_data/esm2_t33_650M_UR50D \
  --label-col label \
  --id-col id \
  --feature-prefix p_ \
  --n-splits 5 \
  --seed 13 \
  --val-size 0.1 \
  --metric cosine \
  --shuffle-ties

python run_biosieve_splits_from_reduction_summary.py \
  --reduction-summary ../../reduced_dataset/not_normalized_data/esmc_300m/reduction_summary.csv \
  --reduction-dir ../../reduced_dataset/not_normalized_data/esmc_300m/ \
  --full-data ../../reduced_dataset/not_normalized_data/esmc_300m/full_data_with_biosieve_id.csv \
  --output-dir ../../split_process/not_normalized_data/esmc_300m \
  --label-col label \
  --id-col id \
  --feature-prefix p_ \
  --n-splits 5 \
  --seed 13 \
  --val-size 0.1 \
  --metric cosine \
  --shuffle-ties

python run_biosieve_splits_from_reduction_summary.py \
  --reduction-summary ../../reduced_dataset/not_normalized_data/mistral-Prot-v1-134M/reduction_summary.csv \
  --reduction-dir ../../reduced_dataset/not_normalized_data/mistral-Prot-v1-134M/ \
  --full-data ../../reduced_dataset/not_normalized_data/mistral-Prot-v1-134M/full_data_with_biosieve_id.csv \
  --output-dir ../../split_process/not_normalized_data/mistral-Prot-v1-134M \
  --label-col label \
  --id-col id \
  --feature-prefix p_ \
  --n-splits 5 \
  --seed 13 \
  --val-size 0.1 \
  --metric cosine \
  --shuffle-ties

#python run_biosieve_splits_from_reduction_summary.py \
#  --reduction-summary ../../reduced_dataset/not_normalized_data/one_hot/reduction_summary.csv \
#  --reduction-dir ../../reduced_dataset/not_normalized_data/one_hot/ \
#  --full-data ../../reduced_dataset/not_normalized_data/one_hot/full_data_with_biosieve_id.csv \
#  --output-dir ../../split_process/not_normalized_data/one_hot \
#  --label-col label \
#  --id-col id \
#  --feature-prefix p_ \
# --n-splits 5 \
#  --seed 13 \
#  --val-size 0.1 \
#  --metric cosine \
#  --shuffle-ties

python run_biosieve_splits_from_reduction_summary.py \
  --reduction-summary ../../reduced_dataset/not_normalized_data/prot_bert/reduction_summary.csv \
  --reduction-dir ../../reduced_dataset/not_normalized_data/prot_bert/ \
  --full-data ../../reduced_dataset/not_normalized_data/prot_bert/full_data_with_biosieve_id.csv \
  --output-dir ../../split_process/not_normalized_data/prot_bert \
  --label-col label \
  --id-col id \
  --feature-prefix p_ \
  --n-splits 5 \
  --seed 13 \
  --val-size 0.1 \
  --metric cosine \
  --shuffle-ties

python run_biosieve_splits_from_reduction_summary.py \
  --reduction-summary ../../reduced_dataset/not_normalized_data/prot_t5_xl_uniref50/reduction_summary.csv \
  --reduction-dir ../../reduced_dataset/not_normalized_data/prot_t5_xl_uniref50/ \
  --full-data ../../reduced_dataset/not_normalized_data/prot_t5_xl_uniref50/full_data_with_biosieve_id.csv \
  --output-dir ../../split_process/not_normalized_data/prot_t5_xl_uniref50 \
  --label-col label \
  --id-col id \
  --feature-prefix p_ \
  --n-splits 5 \
  --seed 13 \
  --val-size 0.1 \
  --metric cosine \
  --shuffle-ties