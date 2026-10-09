# Phase 2 ingestion program

This ingestion program runs a participant's trained out-of-distribution (OoD)
detector on the private Phase 2 test set. It performs inference only: it does
not provide training data and does not call a `fit` method to train the models.

## Participant submission contract

The submission must be a ZIP archive containing `model.py` at its root. The
archive may also contain Python helper modules, packages, model checkpoints,
the weak-lensing mask, and other artifacts needed for inference.

`model.py` must define a parameterless `Model` class with a `predict` method:

```python
class Model:
    def __init__(self):
        # Load checkpoints and other submission artifacts here.
        ...

    def predict(self, test_data):
        # Return one score per test sample. Larger means more likely OoD.
        return ood_scores
```

The ingestion program instantiates `Model()` and calls `predict(test_data)`
once. It does not call any other participant method.

### Test data

`test_data` is a read-only, memory-mapped NumPy array loaded directly from the
private test `.npy` file. Its shape is `(N, 132019)`, where `N` is the number
of test samples, and its dtype is `float16`. The first dimension is always the
sample dimension.

The ingestion program does not load or apply the weak-lensing survey mask and
does not expand the flattened pixels into `(1424, 176)` maps. A model that
needs full maps must include the mask in its submission and load it relative
to `model.py`, for example:

```python
import os
import numpy as np
model_dir = os.path.dirname(os.path.abspath(__file__))
mask = np.load(os.path.join(model_dir, "WIDE12H_bin2_2arcmin_mask.npy"))
```

### Prediction output

`Model.predict` must return a one-dimensional array-like object containing
exactly `N` OoD scores. The scores must be finite, real numeric values. 

The ingestion program rejects outputs that:

- Are not convertible to a NumPy array.
- Are not one-dimensional.
- Do not contain exactly one score per test sample.
- Contain non-numeric or complex values.
- Contain `NaN` or infinite values.

Each output score must depend only on its corresponding test sample. The test
set must not be used collectively for fitting, adaptation, calibration,
normalization, ranking, clustering, or model selection.

## Output files

After successful inference, the ingestion program writes:

- `result.json`, containing `{"ood_scores": [...]}` for the scoring program.
- `ingestion_duration.json`, containing the runtime in minutes.

Participants return only the raw score array from `predict`; the ingestion
program creates the surrounding `ood_scores` dictionary.

## Execution

Codabench invokes the program using the command in `metadata.yaml`:

```bash
python3 run_ingestion.py --codabench
```

In Codabench mode, the directories are:

- Private test data: `/app/data`
- Participant submission: `/app/ingested_program`
- Ingestion output: `/app/output`

For a local run, execute the command below from the `Cosmology_Challenge`
directory:

```bash
python3 ingestion_program/run_ingestion.py
```

The local run uses `input_data`, `sample_code_submission`, and
`sample_result_submission` under the repository root. The test filename is
defined by `TEST_DATA_FILENAME` in `ingestion.py`.
