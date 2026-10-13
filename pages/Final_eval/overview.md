# FAIR Universe - Weak Lensing ML Uncertainty Challenge Phase 2 Final Submission
*** 

### 🚀 Phase 2 Final Submission deadline: 2026/10/20 11:59 PM (AoE)
***
The public phase of the competition has ended. **To be eligible for the final holdout dataset evaluation and competition prizes, participants must complete all the following final submission steps by 2026/10/20 (Tuesday) 11:59 PM AoE**.

1. Join our [**<ins>Slack channel</ins>**](https://join.slack.com/t/fairuniversef-iti7727/shared_invite/zt-4cquy88ic-dSinuZCTWiLw3Tzj9Y2a3Q) so that we can easily communicate anything about the final submission and evaluation (this is optional but strongly recommended).

2. Submit [**<ins>this form</ins>**](https://docs.google.com/forms/d/e/1FAIpQLSdLWOdqE3wo4zdTy1xxUek8rI3yxcWJw0pstar4Usd_0F1pIg/viewform?usp=publish-editor). You will upload: (1) A brief summary write-up of your method (use our [**<ins>LaTeX template</ins>**](https://www.overleaf.com/read/kjznzsbzvmvw#21668e)); (2) A repository of the code you used.

3. **You must use this Codabench page to prepare a model submission file.** 
    
    Please sign up this Codabench page with the Codabench account you used for the public competition phase. This Codabench page enables you to prepare, test, and debug your model submission file for the final evaluation. 
    
    In this new Codabench page, **you will submit ***a model*** (not a prediction).** Codabench will directly run your model using an AWS instance and a predefined Docker image to produce your score. 
    You have to prepare your inference script, saved model checkpoints, and any precomputed files in a simple format we defined (a ZIP file with `model.py` and files for model submission; see below). 

    
    As this Codabench page loads the public test dataset (`WIDE12H_bin2_2arcmin_kappa_test_phase2_new_v2.npy`) from its backend, the score you get on the leaderboard here should be the same as the score of the public submission ID you specified. This will allow you to check if there is any issue in your model submission. After the final submission deadline, we will download your model submission file from this Codabench and use it for the final evaluation. It is your obligation to ensure that the model submission can be run without any issues.
    
    ⚠️ **Very Important notes:**

    - We will only accept your model submission for the holdout dataset evaluation to determine the final ranking if your model submission can be successfully run on this Codabench page and reproduce your public score.

    - The new Codabench page can only be used to test and debug your model submission. It should **NOT** be used for further model development. 


    ## How to prepare the model submission file for the new Codabench page?
    You will submit **a ZIP file with `model.py`** at its root (**not inside an additional folder**) to the new Codabench page. You may include model checkpoints, helper modules, the survey mask, and any other files required for inference. **Please only include the necessary code of your inference pipeline in the submission. No model training is allowed.** In **`model.py`**, define a parameterless class named **`Model`** with a **`predict`** method:

    ```python
    class Model:
        def __init__(self):
        # Load your trained model and other required files here.
        ...

        def predict(self, test_data):
        # Return one OoD score for each test sample.
        return ood_scores
    ```

    The ingestion program will instantiate **`Model()`** and call **`predict(test_data)`** once. `test_data` is a read-only NumPy array with shape `(N, 132019)` and dtype `float16`. Your method must return a one-dimensional array of `N` finite OoD scores. 

    Load all bundled files using paths relative to **`model.py`**. For example, with `os.path.dirname(os.path.abspath(__file__))`. If your method requires full `(1424, 176)` maps, include the mask in your ZIP and reconstruct the maps inside your code.

    Please see the [**<ins>ingestion program and its README</ins>**](https://github.com/FAIR-Universe/Cosmology_Challenge/tree/master/ingestion_program) for the complete interface and local testing instructions. 

    ## Examples of the model submission
    To help you understand the expected format of the model submission, we have prepared the [**<ins>sample code submission</ins>**](https://github.com/FAIR-Universe/Cosmology_Challenge/tree/master/sample_code_submission) and the [**<ins>model submission files of baseline methods</ins>**](https://github.com/FAIR-Universe/Cosmology_Challenge/tree/master/model_submission_baseline). You may download these files, compress any of them into a ZIP file (excluding its parent directory), and submit to this Codabench page.

    🤖 You may use an AI agent to help you adapt your inference code into **`model.py`**, but it is your responsibility to ensure that everything can reproduce your public score.


    ## Docker image for model submission
    See the Dockerfile [**<ins>here</ins>**](https://github.com/FAIR-Universe/Cosmology_Challenge/tree/master/docker). The built Docker image `ayaga3540/fair_universe:v1.0` is available on Docker Hub.

    ## Hardware specs for model submission
    ```
    GPU: 1 × NVIDIA L40S
    GPU Memory: 48 GB GDDR6
    vCPUs: 4 (AMD EPYC processors)
    System Memory (RAM): 32 GiB
    Local Storage: 1 × 250 GB NVMe SSD
    Network Performance: Up to 20 Gbps
    EBS Bandwidth: Up to 5 Gbps
    ```


## Timeline for the final submission
***
- **2026/10/11**: End of the public competition phase
- **2026/10/13—10/20**: Final submission phase
- **2026/10/20**: 
    - Deadline of the final submission phase: Participants must complete the required form and the Codabench model submission
    - Final holdout dataset evaluation: Organizers will download participants' Codabench model submissions and evaluate on the holdout dataset to determine the final rankings
- **2026/10/25**: Announcement of the final rankings



## Contact
***
Visit our website: <ins>https://fair-universe.lbl.gov/</ins>

Email: <ins>fair-universe@lbl.gov</ins>

Updates will be announced through the [<ins>Codabench forum</ins>](https://www.codabench.org/forums/10738/).
