# Evaluation
***
Participants must submit a model to the Codabench platform that can analyze a dataset to determine the point estimates $\hat{\Omega_m}$, $\hat{S_8}$, and uncertainties $\sigma_{\Omega_m}$, $\sigma_{S_8}$. In the second phase, the model should also determine the probability $p$ that the given dataset is consistent with the training data.

- In the **first phase**, the model's performance will be ranked with the following score:
    $$
        -\frac{1}{N_{test}} \sum_{i=0}^{N_{test}} \left(\frac{(\hat{\Omega_m} - \Omega_m^{\textrm{truth}})^2}{\sigma_{\Omega_m}^2}+\frac{(\hat{S_8} - S_8^{\textrm{truth}})^2}{\sigma_{S_8}^2}+\log(\sigma_{\Omega_m}^2) + \log(\sigma_{S_8}^2) + \lambda \[ (\hat{\Omega}_m - \Omega_m^{\textrm{truth}})^2 + (\hat{S}_8 - S_8^{\textrm{truth}})^2 \] \right).
    $$
    The first term corresponds to the Kullback–Leibler (KL) divergence (up to some constants) between the true posterior distribution and the Gaussian distribution with the predicted mean and standard deviation. We expect the posterior distribution to be pretty Gaussian and the correlation between $\Omega_m$ and $S_8$ to be small, thus the Gaussian approximation with diagonal covariance matrix should be good enough. The second term is an MSE loss with weight $\lambda=1000$ to penalize bad point estimates.
    
    For this task, the test data will be drawn from the same distribution as the training data with unknown cosmological parameters and systematics.  

- In the **second phase**, the model's OoD detection performance will be assessed with the following score
    $$
        \frac{1}{N_{test}} \sum_{i=0}^{N} \left(y_i \log(p_i)+(1-y_i)\log(1-p_i)\right),
    $$
    where $y_i=1$ if the dataset is InD, and $y_i=0$ if the dataset is OoD.

    For this task, some of the test data will be generated assuming different physical models (OoD), and the participant model should estimate the probability $p_i$ of whether each test data is drawn from the same distribution as the training data. The participant will not be provided with OoD examples or any information on how the OoD test data are generated.

The final score used to evaluate the model's performace is a weighted sum of the two scores above, where the score in the first phase is only evaluated on InD datasets. 
