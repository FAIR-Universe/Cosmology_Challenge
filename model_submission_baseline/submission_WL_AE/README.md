# WL autoencoder ingestion submission

`model.py` implements the Phase 2 ingestion contract (`Model().predict(test_data)`).
It loads the bundled checkpoint, mask, and in-distribution reconstruction-score
reference. `WL_DEVICE` and `WL_AE_BATCH_SIZE` may be set to override automatic
device selection and the default batch size of 64.
