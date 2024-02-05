import mlflow

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