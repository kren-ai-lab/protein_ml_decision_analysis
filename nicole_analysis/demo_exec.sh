#!/usr/bin/bash

python training_model_external_cv_with_descriptors.py \
  --seed 113 \
  --partition_strategy random_kfold \
  --representation_strategy ankh2 \
  --redundancy_strategy distance_aware \
  --splits_root /home/david/Downloads/nicole_toxica/P43_toxic_peptides/split_data/cytotoxic/ankh2_ext1/p98/split_random_kfold/seed_113 \
  --descriptor_file /home/david/Downloads/nicole_toxica/numerical_representation/data_toxic_processed_ankh2_ext1.csv \
  --join_col sequence \
  --output_dir demo \
  --label_col label \
  --feature_prefix p_ \
  --config ../general_configs/config_hyperparameters_algorithm.json \
  --algorithm LogisticRegression \
  --scaler normalizer_l2
