# Repository instructions

- Never delete, modify, rename, or overwrite the original `.ipynb` or `.xlsx` files.
- Use paths relative to the repository; do not depend on a user's home directory.
- Keep learned preprocessing and the estimator in one saved pipeline artifact.
- Do not duplicate model preprocessing in `app.py`; call `predictor.py`.
- Fit caps, coordinate means, time origins, encoders, and models on training data.
- Read categories and displayed metrics from saved metadata, never hard-code them.
- A valuation must come only from the saved sklearn model.
- Ollama may explain factors but must never calculate, replace, or revise a valuation.
- Preserve validation for missing, negative, invalid, and non-finite inputs.
- Test changes with real rows from `issue_permit_sample.xlsx`.
- Run `python test_prediction.py` after changes.
