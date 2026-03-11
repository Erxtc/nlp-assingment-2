import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

SEED = 10

dataset = load_dataset("sh0416/ag_news")

df_train = pd.DataFrame(dataset["train"])
df_test = pd.DataFrame(dataset["test"])

print("Train samples:", len(df_train))
print("Test samples: ", len(df_test))

# Make dev from train
df_train, df_dev = train_test_split(
    df_train,
    test_size=0.1,
    random_state=SEED,
    stratify=df_train["label"],  # Ensure equally distributed categories across splits
)

print("Train:", len(df_train))
print("Test: ", len(df_test))
print("Dev:  ", len(df_dev))
