import os
os.environ["KERAS_BACKEND"] = "torch"
os.environ["MLFLOW_TRACKING_URI"] = "https://mlflow-server-med-jupyter-central-dev.apps.dev.ocp.bisinfo.org"
import keras
import mlflow
import numpy as np
import pandas as pd
import temporal_fusion_transformers as tft

from model_autoreg import fit_ar_model
from model_tft import build_tft, get_train_val_data, get_default_callbacks
import plot

from mlflow_utils import add_tag_to_active_run, get_or_create_experiment


def main():
    d_model = 8
    dropout_rate = 0.2
    learning_rate = 0.02
    epochs=10
    n_samples = 1000
    start_date_train=pd.Timestamp("1980-01-01")
    end_date_train=pd.Timestamp("2018-01-01")

    train_data, val_data = get_train_val_data(
        n_samples=n_samples,
        n_samples_val=int(0.1*n_samples),
        df_daily_input=tft.df_input_scl,
        df_target=tft.df_target_1m_pct,
        start_date_train=start_date_train,
        end_date_train=end_date_train,
        start_date_val=pd.Timestamp("2018-01-01"),
        end_date_val=pd.Timestamp("2020-01-01"),
        batch_size=64
    )

    experiment_id = get_or_create_experiment("TFT")
    with mlflow.start_run(run_name="Parent run", experiment_id=experiment_id) as run:
        for country in tft.countries:
            with mlflow.start_run(nested="True", run_name=f"Autoreg_{country}", experiment_id=experiment_id):
                autoreg_model = fit_ar_model(tft.df_target_1m_pct, country, start_date_train, end_date_train, lags=12)

        for mode in ["partial", "full"]:
            with mlflow.start_run(nested="True", run_name=f"TFT_{mode}", experiment_id=experiment_id) as tft_run:
                skip_attn = mode == "partial"
                tft_model = build_tft(
                    d_model=d_model, dropout_rate=dropout_rate, learning_rate=learning_rate, partial=skip_attn
                )
                callbacks = get_default_callbacks(tft_run)

                hist = tft_model.fit(
                    train_data,
                    validation_data=val_data,
                    callbacks=callbacks,
                    epochs=epochs
                )
                params = {
                    "d_model": d_model,
                    "dropout_rate": dropout_rate,
                    "learning_rate": learning_rate,
                }
                # Log to MLflow
                mlflow.log_params(params)

                tft_model.save_weights("tft.weights.h5")
                mlflow.log_artifact("tft.weights.h5")
                
                tft_model.save("model.keras")
                mlflow.log_artifact("model.keras")
                country = "US"
                date = tft.dates_train[np.random.randint(0, high=len(tft.dates_train))]
                mlflow.log_figure(plot.plot_mom_change(tft_model, country, date), f"{country}_inflation_in_sample_mom.png")
                mlflow.log_figure(plot.plot_yoy_change(tft_model, country, date), f"{country}_inflation_in_sample_yoy.png")
                


if __name__ == "__main__":
    mlflow.pytorch.autolog()
    mlflow.statsmodels.autolog()
    main()
