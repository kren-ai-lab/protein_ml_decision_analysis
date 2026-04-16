python run_biosieve_reducers_from_percentiles.py \
  --embedding-npy ../../analysed_space/not_normalized_data/ankh2-ext1/artifacts/training_embeddings.npy \
  --percentiles-csv ../../analysed_space/not_normalized_data/ankh2-ext1/tables/ankh2-ext1_similarity_percentiles.csv \
  --input-data ../../represented_dataset/ankh2-ext1/full_data.csv \
  --output-dir ../../reduced_dataset/not_normalized_data/ankh2-ext1 \
  --label-col label \


python run_biosieve_reducers_from_percentiles.py \
  --embedding-npy ../../analysed_space/not_normalized_data/ankh3-large/artifacts/training_embeddings.npy \
  --percentiles-csv ../../analysed_space/not_normalized_data/ankh3-large/tables/ankh3-large_similarity_percentiles.csv \
 --input-data ../../represented_dataset/ankh3-large/full_data.csv \
  --output-dir ../../reduced_dataset/not_normalized_data/ankh3-large \
  --label-col label \

python run_biosieve_reducers_from_percentiles.py \
  --embedding-npy ../../analysed_space/not_normalized_data/esm2_t6_8M_UR50D/artifacts/training_embeddings.npy \
  --percentiles-csv ../../analysed_space/not_normalized_data/esm2_t6_8M_UR50D/tables/esm2_t6_8M_UR50D_similarity_percentiles.csv \
  --input-data ../../represented_dataset/esm2_t6_8M_UR50D/full_data.csv \
  --output-dir ../../reduced_dataset/not_normalized_data/esm2_t6_8M_UR50D \
  --label-col label \


python run_biosieve_reducers_from_percentiles.py \
  --embedding-npy ../../analysed_space/not_normalized_data/esm2_t12_35M_UR50D/artifacts/training_embeddings.npy \
  --percentiles-csv ../../analysed_space/not_normalized_data/esm2_t12_35M_UR50D/tables/esm2_t12_35M_UR50D_similarity_percentiles.csv \
  --input-data ../../represented_dataset/esm2_t12_35M_UR50D/full_data.csv \
  --output-dir ../../reduced_dataset/not_normalized_data/esm2_t12_35M_UR50D \
  --label-col label \

python run_biosieve_reducers_from_percentiles.py \
  --embedding-npy ../../analysed_space/not_normalized_data/esm2_t30_150M_UR50D/artifacts/training_embeddings.npy \
  --percentiles-csv ../../analysed_space/not_normalized_data/esm2_t30_150M_UR50D/tables/esm2_t30_150M_UR50D_similarity_percentiles.csv \
 --input-data ../../represented_dataset/esm2_t30_150M_UR50D/full_data.csv \
  --output-dir ../../reduced_dataset/not_normalized_data/esm2_t30_150M_UR50D \
  --label-col label \


python run_biosieve_reducers_from_percentiles.py \
  --embedding-npy ../../analysed_space/not_normalized_data/esm2_t33_650M_UR50D/artifacts/training_embeddings.npy \
  --percentiles-csv ../../analysed_space/not_normalized_data/esm2_t33_650M_UR50D/tables/esm2_t33_650M_UR50D_similarity_percentiles.csv \
  --input-data ../../represented_dataset/esm2_t33_650M_UR50D/full_data.csv \
  --output-dir ../../reduced_dataset/not_normalized_data/esm2_t33_650M_UR50D \
  --label-col label \


python run_biosieve_reducers_from_percentiles.py \
 --embedding-npy ../../analysed_space/not_normalized_data/esmc_300m/artifacts/training_embeddings.npy \
  --percentiles-csv ../../analysed_space/not_normalized_data/esmc_300m/tables/esmc_300m_similarity_percentiles.csv \
  --input-data ../../represented_dataset/esmc_300m/full_data.csv \
  --output-dir ../../reduced_dataset/not_normalized_data/esmc_300m \
  --label-col label \

python run_biosieve_reducers_from_percentiles.py \
  --embedding-npy ../../analysed_space/not_normalized_data/mistral-Prot-v1-134M/artifacts/training_embeddings.npy \
  --percentiles-csv ../../analysed_space/not_normalized_data/mistral-Prot-v1-134M/tables/mistral-Prot-v1-134M_similarity_percentiles.csv \
  --input-data ../../represented_dataset/mistral-Prot-v1-134M/full_data.csv \
  --output-dir ../../reduced_dataset/not_normalized_data/mistral-Prot-v1-134M \
  --label-col label \

#python run_biosieve_reducers_from_percentiles.py \
#  --embedding-npy ../../analysed_space/not_normalized_data/one_hot/artifacts/training_embeddings.npy \
#  --percentiles-csv ../../analysed_space/not_normalized_data/one_hot/tables/one_hot_similarity_percentiles.csv \
# --input-data ../../represented_dataset/one_hot/full_data.csv \
#  --output-dir ../../reduced_dataset/not_normalized_data/one_hot \
# --label-col label \


python run_biosieve_reducers_from_percentiles.py \
  --embedding-npy ../../analysed_space/not_normalized_data/prot_bert/artifacts/training_embeddings.npy \
  --percentiles-csv ../../analysed_space/not_normalized_data/prot_bert/tables/prot_bert_similarity_percentiles.csv \
  --input-data ../../represented_dataset/prot_bert/full_data.csv \
  --output-dir ../../reduced_dataset/not_normalized_data/prot_bert \
  --label-col label \

python run_biosieve_reducers_from_percentiles.py \
  --embedding-npy ../../analysed_space/not_normalized_data/prot_t5_xl_uniref50/artifacts/training_embeddings.npy \
  --percentiles-csv ../../analysed_space/not_normalized_data/prot_t5_xl_uniref50/tables/prot_t5_xl_uniref50_similarity_percentiles.csv \
  --input-data ../../represented_dataset/prot_t5_xl_uniref50/full_data.csv \
  --output-dir ../../reduced_dataset/not_normalized_data/prot_t5_xl_uniref50 \
  --label-col label \