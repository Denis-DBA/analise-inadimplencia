"""Divisão reproduzível que mantém atributos idênticos no mesmo grupo."""
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def dividir_dados(X, y):
    groups = pd.util.hash_pandas_object(X, index=False)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train, test = next(splitter.split(X, y, groups=groups))
    return X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test]
