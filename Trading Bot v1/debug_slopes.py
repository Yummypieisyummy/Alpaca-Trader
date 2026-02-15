import pandas as pd

df = pd.read_csv('data/nvda_features_19-20.csv')
df['ema750'] = df['close'].ewm(span=750, adjust=False).mean()
df['ema750_slope'] = df['ema750'].diff()

print('Date range:', df['timestamp'].min(), 'to', df['timestamp'].max())
print('Total rows:', len(df))
print('\nEMA750 slope statistics:')
print(df['ema750_slope'].describe())
print('\nMax slope:', df['ema750_slope'].max())
print('Min slope:', df['ema750_slope'].min())
print('\nRows where slope > 0.005:', (df['ema750_slope'] > 0.005).sum())
print('Rows where slope < -0.005:', (df['ema750_slope'] < -0.005).sum())
print('\nRows where slope > 0.001:', (df['ema750_slope'] > 0.001).sum())
print('Rows where slope < -0.001:', (df['ema750_slope'] < -0.001).sum())
