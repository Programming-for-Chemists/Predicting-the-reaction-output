import pandas as pd

df = pd.read_csv("8_SEPT_APPROVED_full_dataset.csv")

class_counts = df["ReactionClass"].value_counts().sort_index()
total_row = 1008
samples_per_class = total_row // len(class_counts)

sampled_df = pd.DataFrame()
for reaction_class in class_counts.index:
    class_subset = df[df["ReactionClass"] == reaction_class]
    class_sample = class_subset.head(min(samples_per_class, len(class_subset)))
    sampled_df = pd.concat([sampled_df, class_sample])

sampled_df.reset_index()
df = sampled_df.copy()
df.to_csv("df-predicting_the_reaction_output.csv", index=False)
