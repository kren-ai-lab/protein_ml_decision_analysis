# A Practical Framework for Reliable Bioinformatics Modeling: A Data-Centric Benchmarking Study in Antioxidant Proteins

## Preliminary steps:
    - Create an environment or just use an available environment
    - Install the library:
        ```
        pip install  -e .
        ```

## Working with preprocessing:

    1. Download the data from: https://drive.google.com/drive/u/1/folders/1OQk0EBxAj88kY6LPx19yKQ47jbrDFGSk
    2. Using the raw data and also download the xlsx file
    3. Check the notebooks and modify the paths as your convenience
    4. Run all notebooks in the folder parsers
    5. Run the notebook in the folder pivoting_data
    6. Run the notebook in the folder preprocessing_and_cleaning
        - In this notebook you can define the lengths and the use of canonical or extended residues

> With this, you will have a processed and cleaned dataset ready for the next steps.

## Working with redundancy:

Once you have obtained the processed dataset, you can apply the redundance reduction strategies. For this, the biosieve library is used as tool for redundancy. However, also you should produce a numerical representation based on our sylphy library if you will be working with redundancy strategies.

So, if you will apply redundancy based on cosine distance, please follow the next steps:

1. Apply the numerical representation
2. Apply the embedding analysis for reductions
3. Apply the redundance based on the identified threshold in the step 2.

Alternatively, you can integrate directly the splitting process by using biosieve. The recomendations is making the splits using the following strategies:

- Random k-fold
- Stratified k-fold
- Distance-aware k-fold

These strategies will help you to produce different levels of dataset. To facilitate the execution and automatization of these steps, we have implemented different pipelines that you can see in the folder pipelines.

So, if you will apply redundance based on homology-based, the idea is very similar. However, the threshold should change based on the processed data. Please, follow the next steps:

1. Apply the homology-based analysis for reductions
2. Apply the redundance based on the identified threshold in the step 1.

If you will integrate directly spliting methods, you can explore the same strategies that in the first option. However, you should follow the next steps:

1. Apply numerical representation
2. Apply spliting based on the selected strategy.

Similar to the first option, we have implemented pipelines for automatizating these steps

> Note: the numerical representation can be used by applying Sylphy library. 

The baseline approach should be:

- Not redundancy process
- One hot as numerical representation 
- Random split 

### Generated pipelines and runing process


## Working with training process

The training process depends of the strategy that you are looking to explore. In this work, we will be working with the following strategies:

- Integrating classic ML approaches: Just exploring traditional supervised learning algorithms with hyperparameter exploring based on grids
- Integrating simple deep learning architectures: Just exploring simple and traditional deep learning architectures with hyperparameter exploring based on grids
- Integrating simple fine-tuning approaches: By applying pre-trained protein language models, fine tuning could be exploring using simple partial freezing or adapters (LoRA)


- Training with single model:

```
python training_model_external_cv.py --seed 13 --partition_strategy random_kfold --representation_strategy esm2_t6 --redundancy_strategy baseline --splits_root ../../pipelines/embedding_random_k_fold/results/split_process/random_kfold_experiment/ --output_dir demo --label_col label --feature_prefix p_ --config ../../general_configs/config_hyperparameters_algorithm.json --algorithm RandomForestClassifier --scaler normalizer_l2
```