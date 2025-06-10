import pandas as pd

# Đọc file processed_data.csv và lấy 300000 dòng đầu
df = pd.read_csv("processed_data.csv")
df_sample = df.head(300000)
df_sample.to_csv("processed_data_300k.csv", index=False)
print("✅ Đã lưu 300000 dòng vào processed_data_300k.csv")