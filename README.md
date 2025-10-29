# Lung Cancer Prediction — Implementation details

This repository contains a small end-to-end machine learning project to classify lung-cancer susceptibility into three levels (for example: low / medium / high) using clinical and behavioral features. The project includes data-preprocessing utilities, a training/testing pipeline, saved model artifacts and a Streamlit demo app for interactive predictions.

This README documents the implementation, code layout, how to run the app, and how to retrain or test models locally.

---

## Streamlit App Link:
[Lung Cancer Susceptibility Predictor](https://lung-cancer-susceptibility-predictor.streamlit.app/)

---

## Repository layout (important files)

- `app/app.py` — Streamlit web app. Loads the pre-fitted column transformer, target encoder and a trained classifier (Random Forest by default) from `models/` and provides a sidebar to enter patient features and request predictions.
- `data/cancer patient data sets.csv` — Raw dataset used for training and experimentation.
- `models/` — Fitted artifacts (joblib files) used by the app and inference scripts:
	- `random_forest_classifier_fitted.joblib` — trained classifier used by the demo app.
	- `col_transformer_fitted.joblib` — column transformer used to scale/encode features.
	- `ordinal_encoder_fitted.joblib` — target encoder used to map Level <-> numeric labels.
	- `logistic_regression_fitted.joblib` — example saved model (used in experiments/tests).
- `notebooks/Lung_Cancer_Prediction.ipynb` — notebook used during exploration and model development.
- `src/utils.py` — helper functions used throughout the project (data loading, model save/load, train/test utilities, generators for test data, feature ranges, etc.).
- `src/preprocessing_n_training_pipeline.py` — script that demonstrates how the pipeline pieces (split, transformer, encoders) are used to produce training-ready data and how models are trained/tested.
- `src/inference.py` — small inference demo that shows how to load artifacts, transform synthetic/test data, and make predictions.

## Implementation details

High-level flow:

1. Data is read from `data/cancer patient data sets.csv` using `src/utils.load_data` which drops `Patient Id` and `Gender` columns and returns a pandas DataFrame.
2. `src/preprocessing_n_training_pipeline.preprocess_data` performs a train/test split (25% test) using `split_data` and then:
	 - Loads a fitted `col_transformer` from `models/col_transformer_fitted.joblib` and transforms train & test features.
	 - Loads an `ordinal_encoder` from `models/ordinal_encoder_fitted.joblib` and encodes the `Level` column (target) for train & test.
	 - Returns transformed x_train, x_test, y_train, y_test.
3. `src/preprocessing_n_training_pipeline.train_model` and `test_model` are thin wrappers around scikit-learn model fit/score.
4. Models and preprocessing artifacts are saved/loaded with `joblib` via `src/utils.save_model` / `src/utils.load_model`. These helper functions resolve relative paths against the current working directory and require the target filename to end with `.joblib`.
5. The Streamlit app (`app/app.py`) loads the saved artifacts via full paths (in the provided code it loads models using an absolute path: `A:/AI-Projects/personal-projects/lung-cancer-classification/models/...`). The app:
	 - Presents sliders for each feature in the sidebar.
	 - Builds a single-row DataFrame with the user inputs.
	 - Transforms inputs with the loaded `col_transformer`, predicts with the classifier, then decodes the predicted numeric label to its string representation with the `ordinal_encoder`.
	 - Shows results with color-coded Streamlit messages (success/warning/error based on predicted level).

Key helper functions in `src/utils.py` (summary):

- `load_data(data_path: str)` — reads CSV at `data_path` using pandas, drops `Patient Id` and `Gender`, returns DataFrame. Raises ValueError for missing file or bad extension.
- `split_data(df)` — splits into X and Y and calls `train_test_split(test_size=0.25, random_state=42)`.
- `load_model(model_path: str)` — loads a `.joblib` file with joblib.load and returns the object.
- `save_model(model_object, model_path: str)` — dumps an object to a `.joblib` file using joblib.dump.
- `preprocess_data(df, col_transformer_path, target_encoder_path)` — runs split_data, transforms features with a fitted column transformer and encodes the target with the ordinal encoder (returns transformed arrays / DataFrames ready for training).
- `get_test_data()` — generates a single-row random sample (pandas DataFrame) with plausible ranges for each feature (useful for quick inference/demo testing).

Notes about model artifacts and naming:

- The code expects `.joblib` files under `models/` and checks for the `.joblib` extension. If you change filenames, update paths in `app/app.py` or the scripts.

## How to run the Streamlit demo (local)

Prerequisites

- Python 3.8+ installed.
- Recommended: create a virtualenv and install dependencies listed in `requirements.txt`.

Install dependencies (PowerShell):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

Run the app from the repository root (PowerShell):

```powershell
cd A:\AI-Projects\personal-projects\lung-cancer-classification
streamlit run app\app.py
```

## Training a new model (quick guide)

The repository includes a simple training workflow in `src/preprocessing_n_training_pipeline.py` and an example usage in the top of that file. Steps to retrain:

1. Prepare your environment and install requirements.
2. Load the CSV and inspect the data: `src/utils.load_data('data/cancer patient data sets.csv')`.
3. Use `preprocess_data` to obtain transformed X and encoded y. Note: `preprocess_data` expects fitted `col_transformer` and `ordinal_encoder` artifacts — if you don't have them, you need to build them as part of a preprocessing script (not included here) or modify `preprocess_data` to fit new transformers.
4. Train a scikit-learn estimator, e.g. RandomForestClassifier or LogisticRegression, call `train_model(model, x_train, y_train)`, then `save_model(model, 'models/<name>.joblib')`.
5. Evaluate with `test_model(model, x_test, y_test)`.

Because the column transformer and ordinal encoder are loaded from saved artifacts, re-training end-to-end requires either:

- the original preprocessing script that created and saved `col_transformer_fitted.joblib` and `ordinal_encoder_fitted.joblib` (not included), or
- modify `src/preprocessing_n_training_pipeline.py` to fit new transformers and then save them with `save_model`.

If you want help adding a full preprocessing script that fits and saves these artifacts, I can generate one.

## Inference from command line (script)

`src/inference.py` demonstrates how to:

- load the pre-fitted `col_transformer` & `ordinal_encoder` and a classifier from `models/` using `load_model`,
- create or fetch a test DataFrame (via `get_test_data()`),
- transform the test data with the column transformer, predict with the classifier, and then inverse-transform the numeric prediction to a human-readable label.

Example (run from project root):

```powershell
python -c "from src.inference import *; print('See src/inference.py for usage')"
```

Or run the small demonstration at the top of `src/inference.py` (it prints a sample test row and the predicted label).

## Dependencies

See `requirements.txt` for the exact pins, but the main libraries used are:

- pandas
- numpy
- scikit-learn
- joblib
- streamlit (for the demo app)

## Developer notes & gotchas

- Paths: `src/utils.load_model` and the Streamlit app assume paths are resolved relative to the current working directory. Running code from a different cwd may cause `ValueError: Invalid path or file extension!` — prefer running scripts from the project root or update the paths to absolute locations.
- Model artifacts must have `.joblib` extension — the loader validates this and will raise on mismatch.
- The project currently ships with pre-fitted artifacts in `models/` so the demo app should work out-of-the-box if you run it from the repository root (after fixing the absolute paths in `app/app.py` or running from the same A: location used by the developer).

## Next steps (recommended)

- Make `app/app.py` load models by relative path so the app works regardless of where the repo is cloned.
- Add a small script to (re)create and save preprocessing artifacts (`col_transformer` and `ordinal_encoder`) so the full train-from-scratch flow works.
- Add unit tests for `src/utils` functions (load/save, split_data) and for end-to-end inference.

## Contact / License

This project is licensed under the included `LICENSE` file. For questions, open an issue or contact the repository owner.

