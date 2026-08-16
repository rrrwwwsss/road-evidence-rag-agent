import pandas as pd

df = pd.read_csv('data/要素库.csv')

# 加 errors='coerce'：非法值自动变为 NaT，不会报错
df['发生时间'] = pd.to_datetime(df['发生时间'], format="%Y%m%d_%H%M%S", errors='coerce')

# 格式化输出（NaT 会自动变为空字符串，不会产生异常）
df['发生时间'] = df['发生时间'].dt.strftime("%Y-%m-%d %H:%M:%S")

print(df['发生时间'])
df.to_csv('data/external/records.csv', index=False)  # 通常不需要保存行索引