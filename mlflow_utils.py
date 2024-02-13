import keras
import mlflow
from keras.models import Model
from mlflow import pyfunc
import temporal_fusion_transformers as tft


class KerasModelWrapper(pyfunc.PythonModel):
    def __init__(self, model: Model):
        self.model = model

    def load_context(self, context):
        """Loads the model from the artifact path"""
        self.model = keras.models.load_model(context.artifacts["model"])
        self._hack_build()
        self.model.load_weights(context.artifacts["weights"])

    def predict(self, context, model_input):
        return self.model.predict(model_input)

    def _hack_build(self):
        x, _ = tft.prepare_data_samples(
            n_samples=1,
            df_daily_input=tft.df_input_scl,
            df_target=tft.df_target_1m_pct,
            sampled_day="2000-01-01",
            min_context=365,
            context_length=365,
            country="US",
        )
        self.model(x)


def log_keras_model(model: Model, artifact_path: str):
    """Log Keras model to MLflow as an artifact with custom serialization.

    Args:
        model (Model): The model to log.
        artifact_path (str): The artifact path in MLflow.
    """
    model.save_weights("tft.weights.h5")
    model.save("model.keras")
    mlflow.pyfunc.log_model(
        artifact_path=artifact_path,
        python_model=KerasModelWrapper(None),
        artifacts={"model": "model.keras", "weights": "tft.weights.h5"},
    )


def load_keras_model(model_uri: str) -> KerasModelWrapper:
    """Load a Keras model from MLflow.

    Args:
        model_uri (str): The URI to the logged model in MLflow.

    Returns:
        KerasModelWrapper: The loaded model.
    """
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    return loaded_model


def add_tag_to_active_run(key: str, value: str):
    """
    Add a tag to the active MLflow run.

    Args:
    key (str): The key of the tag.
    value (str): The value of the tag.
    """
    if mlflow.active_run() is None:
        raise RuntimeError("No active run. Ensure you are within an active run context.")

    run_id = mlflow.active_run().info.run_id
    client = mlflow.tracking.MlflowClient()
    client.set_tag(run_id, key, value)


def get_or_create_experiment(experiment_name: str) -> str:
    """
    Retrieve the ID of an existing MLflow experiment or create a new one if it doesn't exist.

    This function checks if an experiment with the given name exists within MLflow.
    If it does, the function returns its ID. If not, it creates a new experiment
    with the provided name and returns its ID.

    Parameters:
    - experiment_name (str): Name of the MLflow experiment.

    Returns:
    - str: ID of the existing or newly created MLflow experiment.
    """

    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)
