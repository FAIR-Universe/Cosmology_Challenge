# WL power-spectrum ingestion submission

`model.py` implements `Model().predict(test_data)`. It loads the bundled mask,
cosmology grid, power-spectrum mean/covariance tables, and in-distribution score
reference. The numerical calculation follows the corresponding Phase 2 notebook.
