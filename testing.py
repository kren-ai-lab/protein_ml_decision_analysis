import pandas as pd
from sylphy.embedding_extractor import create_embedding

df = pd.read_csv("pipelines/data/sequences.csv")

embedder = create_embedding(
    model_name="facebook/esm2_t6_8M_UR50D",
    dataset=df,
    column_seq="sequence",
    name_device="cuda",
    precision="fp32"
)

embedder.run_process(batch_size=8, pool="mean")

print(embedder.coded_dataset)