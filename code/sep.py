import pandas as pd
from sklearn.model_selection import train_test_split

# CSVファイルを読み込む
df = pd.read_csv('file_name.csv', encoding='UTF-8', sep=';')

# データをシャッフルする場合（オプション）
df = df.sample(frac=1, random_state=42)  # ランダムな並び替え。random_stateは乱数生成のシードです。

# 教師データとテストデータに分割（7:3の割合）
train_data, test_data = train_test_split(df, test_size=0.3, random_state=42)

# 新しいCSVファイルとして保存
train_data.to_csv('train_name.csv', index=False)
test_data.to_csv('test_name.csv', index=False)
