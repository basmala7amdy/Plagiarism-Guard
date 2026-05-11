import pandas as pd

df = pd.read_csv(
    "preprocessing/msr_paraphrase_test.txt",
    sep="\t",
    on_bad_lines="skip"  # skip malformed TSV rows
)

print(df.head())