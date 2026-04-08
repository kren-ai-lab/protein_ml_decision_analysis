from sylphy.sequence_encoder import create_encoder
from sylphy.embedding_extractor import create_embedding

class EncodeSequences:

    @classmethod
    def encode_dataset(
        cls, 
        df, 
        column_with_sequence, 
        method="one_hot",
        max_length=1024,
        name_property="ANDN920101"
    ):
        
        if method not in ["one_hot", "ordinal", "kmers", "frequency", "physicochemical", "fft"]:
            raise ValueError("The method should be one of the following options: one_hot, ordinal, kmers, frequency, physicochemical, fft")
        
        if method in ["physicochemical", "fft"]:
            encoder = create_encoder(
                method,  
                dataset=df,
                sequence_column=column_with_sequence,
                max_length=max_length,
                name_property=name_property
            )

        else:
            encoder = create_encoder(
                method,  
                dataset=df,
                sequence_column=column_with_sequence,
                max_length=max_length
            )
        
        encoder.run_process()
        return encoder.coded_dataset
    
    @classmethod
    def get_embeddings(
        cls, 
        df, 
        column_with_sequence, 
        plm_model="facebook/esm2_t6_8M_UR50D",
        name_device="cuda",
        precision="fp32",
        batch_size=8, 
        pool="mean"
    ):
        
        embedder = create_embedding(
            model_name=plm_model,
            dataset=df,
            column_seq=column_with_sequence,
            name_device=name_device,
            precision=precision
        )

        embedder.run_process(
            batch_size=batch_size, 
            pool=pool)

        return embedder.coded_dataset

