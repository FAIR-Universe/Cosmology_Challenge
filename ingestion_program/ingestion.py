# ------------------------------------------
# Imports
# ------------------------------------------
import os
import json
import numpy as np
from datetime import datetime as dt


class Ingestion:
    """
    Class for handling the ingestion process.

    Args:
        None

    Attributes:
        * start_time (datetime): The start time of the ingestion process.
        * end_time (datetime): The end time of the ingestion process.
        * model (object): The model object.
        * test_data (ndarray): The test data array.
        * ood_scores (ndarray): The out-of-distribution scores array.
    """

    def __init__(self):
        """
        Initialize the Ingestion class.
        """
        self.start_time = None
        self.end_time = None
        self.model = None
        self.test_data = None
        self.ood_scores = None

    def start_timer(self):
        """
        Start the timer for the ingestion process.
        """
        self.start_time = dt.now()

    def stop_timer(self):
        """
        Stop the timer for the ingestion process.
        """
        self.end_time = dt.now()

    def get_duration(self):
        """
        Get the duration of the ingestion process.

        Returns:
            timedelta: The duration of the ingestion process.
        """
        if self.start_time is None:
            print("[-] Timer was never started. Returning None")
            return None

        if self.end_time is None:
            print("[-] Timer was never stopped. Returning None")
            return None

        return self.end_time - self.start_time

    def save_duration(self, output_dir=None):
        """
        Save the duration of the ingestion process to a file.

        Args:
            output_dir (str): The output directory to save the duration file.
        """
        duration = self.get_duration()
        duration_in_mins = int(duration.total_seconds() / 60)
        duration_file = os.path.join(output_dir, "ingestion_duration.json")
        if duration is not None:
            with open(duration_file, "w") as f:
                f.write(json.dumps({"ingestion_duration": duration_in_mins}, indent=4))

    def load_test_data(self, input_dir, data_file_name):
        """
        Load the test data.

        """
        print("[*] Loading Test data")

        test_data_file = os.path.join(input_dir, data_file_name)

        print("[*] Loading Test data")

        self.test_data = np.load(test_data_file, mmap_mode="r", allow_pickle=False)

    def init_submission(self, Model):
        """
        Initialize the submitted model.

        Args:
            Model (object): The model class.
        """
        print("[*] Initializing Submmited Model")

        self.model = Model()

    def predict_submission(self):
        """
        Make predictions using the submitted model and validate its output.
        """
        print("[*] Calling predict method of submitted model")
        ood_scores = self.model.predict(self.test_data)
        self.ood_scores = self._validate_ood_scores(ood_scores)

    def _validate_ood_scores(self, ood_scores):
        """
        Validate and normalize the output returned by the submitted model.

        The model must return one finite, real-valued numeric score for every
        test sample.

        Args:
            ood_scores: One-dimensional array-like object returned by
                ``Model.predict``.

        Returns:
            np.ndarray: Validated scores with dtype float64.
        """
        try:
            scores = np.asarray(ood_scores)
        except Exception as error:
            raise TypeError(
                "Model.predict must return a one-dimensional numeric array-like "
                "object"
            ) from error

        if scores.ndim != 1:
            raise ValueError(
                "Model.predict must return a one-dimensional array of OoD "
                f"scores; received shape {scores.shape}"
            )

        expected_size = len(self.test_data)
        if len(scores) != expected_size:
            raise ValueError(
                f"Model.predict must return {expected_size} OoD scores; "
                f"received {len(scores)}"
            )

        if not np.issubdtype(scores.dtype, np.number) or np.issubdtype(
            scores.dtype, np.complexfloating
        ):
            raise TypeError(
                "Model.predict must return real numeric OoD scores; "
                f"received dtype {scores.dtype}"
            )

        scores = scores.astype(np.float64, copy=False)
        if not np.all(np.isfinite(scores)):
            raise ValueError("Model.predict returned NaN or infinite OoD scores")

        return scores

    def compute_result(self):
        """
        Compute the ingestion result.
        """
        print("[*] Computing Ingestion Result")

        self.ingestion_result = {
            "ood_scores": self.ood_scores.tolist()
        }

    def save_result(self, output_dir=None):
        """
        Save the ingestion result to files.

        Args:
            output_dir (str): The output directory to save the result files.
        """
        print("[*] Saving Ingestion Result")
        result_file = os.path.join(output_dir, "result.json")
        with open(result_file, "w") as f:
            f.write(json.dumps(self.ingestion_result, indent=4))
