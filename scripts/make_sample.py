from pathlib import Path
import pandas as pd

raw_path = Path("data/raw/data_base.csv")
out_path = Path("data/sample/sample.csv")

df = pd.read_csv(raw_path)
df_sample = df.sample(n=min(200, len(df)), random_state=42)

out_path.parent.mkdir(parents=True, exist_ok=True)
df_sample.to_csv(out_path, index=False, encoding="utf-8")

print("Saved sample:", out_path, "rows:", len(df_sample))
