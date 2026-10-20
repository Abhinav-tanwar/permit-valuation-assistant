# Permit Valuation Assistant

Saved-model permit valuation with validated inputs and optional constrained explanations.

**Author and maintainer:** Abhinav Tanwar.

## Run

Requires Python 3.12. Install dependencies with `python -m pip install -r requirements.txt`.

```sh
python analysis.py
python -m unittest discover -s tests
```

Also run `python test_prediction.py` for the original real-row integrity checks. Launch the interface with `streamlit run app.py`. Ollama is optional; without it, explanations use the deterministic fallback. The model and all learned preprocessing are saved together in `artifacts/permit_valuation_pipeline.joblib`.

## Methods and limitations

Consolidates the repeated permit notebooks and supplied app. Original workbook/notebook copies are byte-preserved locally and excluded from publication; a reduced, explicitly authorized CSV supports execution. The full saved sklearn pipeline computes every valuation. Caps, coordinate means, time origin, categories, and ranges are trained on the training split. Optional Ollama factor explanations cannot change the returned valuation and fall back when numeric or spelled-out currency language is detected. Evaluation retains the source 70/30 random split and the training-capped log1p target. Log-scale R² and errors are not dollar accuracy; dollar errors refer to the capped target. The sample has few distinct categories/dates/locations and is not evidence of future appraisal performance. The original log-target MAPE is a compatibility diagnostic and is not used as a dollar percentage-error claim.

## Measured results

- **Training:** r2 = 0.7813.
- **Test:** r2 = 0.7803.

[Full calculated metrics](results/metrics.json) · [Input provenance](data/README.md) · [Method changes](docs/methodology.md). Open `report.html` for the visual summary.

The clean notebook in `notebooks/` delegates to the same module. Notebook cells were syntax-checked with outputs cleared; native scripts and tests were executed.
