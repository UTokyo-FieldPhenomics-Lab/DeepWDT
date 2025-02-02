class NoOpRun:
    """A dummy MLflow run object for when MLflow is disabled."""
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        pass

class NoOpMLflow:
    """A dummy MLflow wrapper for when MLflow is not installed or disabled."""
    def __init__(self, run_name):
        self.run_name = run_name
    @staticmethod
    def start_run(run_name=None):
        print(f"[NoOpMLflow] Starting a run with name: {run_name}")
        return NoOpRun()
    @staticmethod
    def set_experiment(name):
        print(f"[NoOpMLflow] Setting experiment: {name}")
    @staticmethod
    def log_artifact(config):
        pass