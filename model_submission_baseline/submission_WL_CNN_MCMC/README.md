# WL CNN + MCMC ingestion submission

`model.py` implements `Model().predict(test_data)`. It bundles the trained CNN,
training-label scaling data, cosmology grid, SciPy interpolators, mask, and
in-distribution score reference. Defaults match the notebook's 10,000 MCMC
steps. `WL_DEVICE`, `WL_CNN_BATCH_SIZE`, and `WL_MCMC_STEPS` are optional runtime
overrides; reducing the MCMC steps changes the baseline calculation.
