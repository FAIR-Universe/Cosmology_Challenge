# ------------------------------------------
# Imports
# ------------------------------------------
import os
import json
import numpy as np
from datetime import datetime as dt
from sklearn.metrics import roc_curve


class Scoring:
    """
    This class is used to compute the scores for the competition.

    Atributes:
        * start_time (datetime): The start time of the scoring process.
        * end_time (datetime): The end time of the scoring process.
        * reference_data (dict): The reference data.
        * ingestion_result (dict): The ingestion result.
        * ingestion_duration (float): The ingestion duration.
        * scores_dict (dict): The scores dictionary.
    """

    def __init__(self, name=""):
        # Initialize class variables
        self.start_time = None
        self.end_time = None
        self.reference_data = None
        self.ingestion_result = None
        self.ingestion_duration = None
        self.scores_dict = {}

    def start_timer(self):
        self.start_time = dt.now()

    def stop_timer(self):
        self.end_time = dt.now()

    def get_duration(self):
        if self.start_time is None:
            print("[-] Timer was never started. Returning None")
            return None

        if self.end_time is None:
            print("[-] Timer was never stoped. Returning None")
            return None

        return self.end_time - self.start_time

    def load_reference_data(self, reference_dir):
        """
        Load the reference data.

        Args:
            reference_dir (str): The reference data directory name.
        """
        print("[*] Reading reference data")
        reference_data_file = os.path.join(reference_dir, "label_test_phase2.npy")  # An array of 0 (InD) and 1 (OoD)
        self.reference_data = np.load(reference_data_file)

    def load_ingestion_result(self, predictions_dir):
        """
        Load the ingestion result.

        Args:
            predictions_dir (str): The predictions directory name.
        """
        print("[*] Reading ingestion result")

        ingestion_result_file = os.path.join(predictions_dir, "result.json")
        with open(ingestion_result_file) as f:
            self.ingestion_result = json.load(f)

    def check_result(self):
        if "ood_scores" not in self.ingestion_result:
            raise KeyError("[-] OoD scores not found in the result!")

        test_set_size = self.reference_data.shape[0]

        if test_set_size != len(self.ingestion_result["ood_scores"]):
            raise ValueError(f"[-] Number of samples in ood_scores should be {test_set_size}")

    def compute_scores(self):
        """
        Compute the scores for the competition.

        """

        # Compute the score for Phase 2.
        # The rank on the leaderboard is sorted by this.
        def _score_phase2(test_labels, ood_scores):
            """
            Computes the ROC-AUC score for Phase 2 based on true InD/OoD labels.

            Parameters
            ----------
            test_labels: np.ndarray
                Array of the ground truths (0=InD, 1=OoD)
            ood_scores: np.ndarray
                Array of continuous test scores that will increase when a sample is more likely to be OoD

            Returns
            -------
            np.ndarray
                ....
            """
            min_fpr = 0.001
            max_fpr = 0.05

            fpr, tpr, thresholds = roc_curve(test_labels, ood_scores)
            fpr_log_interval = np.logspace(np.log10(min_fpr), np.log10(max_fpr), 100)
            tpr_log_interval = np.interp(fpr_log_interval, fpr, tpr)
            score_phase2 = np.mean(tpr_log_interval)

            return fpr_log_interval, tpr_log_interval, score_phase2

        fpr_log_interval, tpr_log_interval, score_phase2 = _score_phase2(self.reference_data, self.ingestion_result["ood_scores"])

        self.scores_dict = {
            "Score": score_phase2,
        }

    def write_scores(self, output_dir):

        print("[*] Writing scores")
        score_file = os.path.join(output_dir, "scores.json")
        with open(score_file, "w") as f_score:
            f_score.write(json.dumps(self.scores_dict, indent=4))
