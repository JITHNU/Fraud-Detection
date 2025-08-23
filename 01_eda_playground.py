import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


CSV = Path(__file__).resolve().parents[1] / 'data' / 'transactions.csv'
df = pd.read_csv(CSV)
print(df.head())
print("\nShape:", df.shape)
print("\nColumns:", list(df.columns))


# Basic stats
print("\nType counts:\n", df['type'].value_counts())
print("\nFraud ratio:")
print(df['isFraud'].value_counts(normalize=True))


# Plot class balance
ax = df['isFraud'].value_counts().sort_index().plot(kind='bar')
ax.set_title('Class Balance (0=legit, 1=fraud)')
ax.set_xlabel('isFraud')
ax.set_ylabel('count')
plt.tight_layout(); plt.show()

ax = df['amount'].clip(upper=df['amount'].quantile(0.99)).hist(bins=50)
ax.set_title('Amount Distribution (clipped 99th)')
ax.set_xlabel('amount')
plt.tight_layout(); plt.show()