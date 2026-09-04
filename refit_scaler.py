import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

df_feat     = pd.read_csv('df_feat.csv')
feature_cols = joblib.load('feature_cols.joblib')

feature_cols_avail = [c for c in feature_cols if c in df_feat.columns]
X = df_feat[feature_cols_avail].fillna(df_feat[feature_cols_avail].median())

scaler = StandardScaler()
scaler.fit(X)

joblib.dump(scaler, 'scaler.joblib')
print("Done. Scaler fitted and saved.")
print("Features used:", feature_cols_avail)