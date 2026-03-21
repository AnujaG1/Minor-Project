import pandas as pd

files = [
    'results/training_data.csv',
    'results/data_1.csv'
    'results/data_2.csv'
]

print(f'Combining {len(files)} files...')
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

# Shuffle
df = df.sample(frac=1, random_state=42)

# Save back to training_data.csv
df.to_csv('results/training_data.csv', index=False)

print(f'Total rows : {len(df)}')
print(f'Attacks    : {df["is_attacker"].sum()}')
print(f'Normal     : {(df["is_attacker"]==0).sum()}')
print(f'Columns    : {list(df.columns)}')