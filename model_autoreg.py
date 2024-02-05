import pandas as pd
from statsmodels.tsa.ar_model import AutoReg

def fit_ar_model(df_target: pd.DataFrame, country: str, start_date: str, end_date: str, lags: int) -> AutoReg:
    # drop the first row of nans from calculating pct_changes
    df_target=df_target.dropna()
    model = AutoReg(df_target.loc[start_date:end_date, country], lags)
    result = model.fit()
    return model