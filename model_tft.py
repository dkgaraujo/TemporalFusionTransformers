import keras
import mlflow
from mlflow.entities.run import Run
from typing import Union

from torch.utils.data import DataLoader
import pandas as pd
import temporal_fusion_transformers as tft


def build_tft(
    d_model: int,
    dropout_rate: float,
    learning_rate: float,
    partial: bool
) ->tft.TFT:
    """_summary_

    Args:
        d_model (int): _description_
        dropout_rate (float): _description_
        learning_rate (float): _description_

    Returns:
        tft.TFT: _description_
    """
    model = tft.TFT(
        d_model=d_model,
        output_size=12,
        dropout_rate=dropout_rate,
        quantiles=tft.quantiles,
        name="tft",
        skip_attention=partial
    )

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=tft.quantile_loss)
    return model

def get_default_callbacks(run: Run) -> list[keras.callbacks.Callback]:
    """_summary_

    Args:
        run (Run): _description_

    Returns:
        list[keras.callbacks.Callback]: _description_
    """
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.2,
        patience=5, min_lr=0.001
    )
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10
    )
    mlflow_log = mlflow.keras_core.MLflowCallback(run, log_every_epoch=True)
    return [
        reduce_lr,
        early_stop,
        mlflow_log
    ]



def get_train_val_data(
    n_samples: int,
    n_samples_val: int,
    df_daily_input: pd.DataFrame,
    df_target: pd.DataFrame,
    start_date_train: pd.Timestamp,
    end_date_train: pd.Timestamp,
    start_date_val: pd.Timestamp,
    end_date_val: pd.Timestamp,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    """_summary_

    Args:
        n_samples (int): _description_
        n_samples_val (int): _description_
        df_daily_input (pd.DataFrame): _description_
        df_target (pd.DataFrame): _description_
        start_date_train (pd.Timestamp): _description_
        end_date_train (pd.Timestamp): _description_
        start_date_val (pd.Timestamp): _description_
        end_date_val (pd.Timestamp): _description_
        batch_size (int): _description_

    Returns:
        tuple[DataLoader, DataLoader]: _description_
    """
    train_data_loader = DataLoader(
        tft.NowcastingData(
            n_samples=n_samples,
            df_daily_input=df_daily_input,
            df_target=df_target,
            start_date=start_date_train,
            end_date=end_date_train,
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    val_data_loader = DataLoader(
        tft.NowcastingData(
            n_samples=n_samples_val,
            df_daily_input=df_daily_input,
            df_target=df_target,
            start_date=start_date_val,
            end_date=end_date_val,
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    return train_data_loader, val_data_loader
