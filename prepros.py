import pandas as pd

# df = pd.read_json("preprocessing/arxiv_ai_subset.json", lines=True)
df = pd.read_csv(
    "preprocessing/msr_paraphrase_test.txt",
    sep="\t",
    on_bad_lines="skip"
)

print(df.head())