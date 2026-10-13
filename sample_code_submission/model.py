# ------------------------------
# Dummy Model Submission
# This is a dummy model for the participants to understand the structure of the code i.e.
# - required functions in the Model class
# - inputs and outputs of the functions
# ------------------------------

# import all necessary libraries or helper functions here
from __future__ import annotations
import numpy as np


class Model:
    """
    This is a model class to be submitted by the participants in their submission.

    This class should consists of the following functions
    1) init
    2) predict:
        - Takes 1 argument: test_data
        - Can be used to get predictions of the test data
        - Returns an array of OoD scores

    Note:   Add more methods if needed e.g. compute test statistics, load pre-trained model etc.
            It is the participant's responsibility to make sure that the submission
            class is named "Model" and that its constructor arguments remains the same.
            The ingestion program initializes the Model class and calls the predict method

            When you add another file with the submission model e.g. a trained model to be loaded and used,
            load it in the following way:

            # Get to the model directory (your submission directory)
            model_dir = os.path.dirname(os.path.abspath(__file__))

            Your trained model file is now in model_dir, you can load it from here

            # Set the device to GPU if available
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    """

    def __init__(self) -> None:
        """
        Model class constructor
        """
        pass

    def predict(self, test_data: np.ndarray) -> np.ndarray:
        """
        Params:
            test_data (np.ndarray): An array containing the test data
                                    shape = (N, 132019), where N is the number of samples in the test dataset

        Functionality:
            This function can be used for predictions using the test sets

            To apply the mask and produce 2D maps, you can include the mask .npy file in model_dir (the submission folder) and load it here.
            # Example:
                mask = np.load(os.path.join(model_dir, "WIDE12H_bin2_2arcmin_mask.npy"))
                N = len(test_data)
                test_maps = np.zeros((N, 1424, 176), dtype=np.float16)
                test_maps[:, mask] = test_data

        Returns:
            A numpy array of the predicted OoD scores
            shape = (N,)
        """

        # Dummy OoD score:
        # scores should be a 1D array of length N, where N is the number of samples in test_data
        scores = test_data[:, 0]
        return scores
