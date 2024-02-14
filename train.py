import os

os.environ["KERAS_BACKEND"] = "torch"
os.environ["MLFLOW_TRACKING_URI"] = (
    "https://mlflow-server-med-jupyter-central-dev.apps.dev.ocp.bisinfo.org"
)
import keras
import mlflow
import numpy as np
import pandas as pd
import temporal_fusion_transformers as tft

from model_autoreg import fit_ar_model
from model_tft import build_tft, get_train_val_data, get_default_callbacks
import plot

import mlflow_utils


def main():
    d_model = 256
    dropout_rate = 0.2
    learning_rate = 0.005
    epochs = 50
    n_samples = 1000
    start_date_train = pd.Timestamp("1980-01-01")
    end_date_train = pd.Timestamp("2018-01-01")

    train_data, val_data = get_train_val_data(
        n_samples=n_samples,
        n_samples_val=int(0.1 * n_samples),
        df_daily_input=tft.df_input_scl,
        df_target=tft.df_target_1m_pct,
        start_date_train=start_date_train,
        end_date_train=end_date_train,
        start_date_val=pd.Timestamp("2018-01-01"),
        end_date_val=pd.Timestamp("2020-01-01"),
        batch_size=64,
    )

    experiment_id = mlflow_utils.get_or_create_experiment("TFT")
    with mlflow.start_run(run_name="Parent run", experiment_id=experiment_id) as run:
        for country in tft.countries:
            with mlflow.start_run(
                nested="True",
                run_name=f"Autoreg_{country}",
                experiment_id=experiment_id,
            ):
                autoreg_data = tft.df_target_1m_pct
                autoreg_model = fit_ar_model(
                    autoreg_data, country, start_date_train, end_date_train, lags=12
                )
        for mode in ["full"]:
            with mlflow.start_run(
                nested="True", run_name=f"TFT_{mode}", experiment_id=experiment_id
            ) as tft_run:
                skip_attn = mode == "partial"
                tft_model = build_tft(
                    d_model=d_model,
                    dropout_rate=dropout_rate,
                    learning_rate=learning_rate,
                    partial=skip_attn,
                )
                callbacks = get_default_callbacks(tft_run)

                hist = tft_model.fit(
                    train_data,
                    validation_data=val_data,
                    callbacks=callbacks,
                    epochs=epochs,
                )
                params = {
                    "d_model": d_model,
                    "dropout_rate": dropout_rate,
                    "learning_rate": learning_rate,
                }
                # Log to MLflow
                mlflow.log_params(params)

                mlflow_utils.log_keras_model(tft_model, "model")


if __name__ == "__main__":
    mlflow.pytorch.autolog()
    mlflow.statsmodels.autolog()
    main()
