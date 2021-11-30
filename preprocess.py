import pandas as pd
import numpy as np

bureau_features = pd.read_csv('./home-credit/features/bureau_features.csv')
bureau_features = bureau_features.fillna(0)

bureau_features.to_csv('./home-credit/features/bureau_features.csv', index=False)