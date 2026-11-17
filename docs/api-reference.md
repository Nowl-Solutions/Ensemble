# Python API reference

These examples describe the functions in this repository. Run Python from the
repository root so that imports such as `from bagging import fit_bagging` work.
The project uses local modules rather than an installed Python package.

## Environment

Install the dependencies in [requirements.txt](../requirements.txt) into a
virtual environment. On Windows PowerShell, activation is optional:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

On macOS or Linux, use `python3 -m venv .venv` and
`.venv/bin/python` for the subsequent commands. The examples below assume
that the selected interpreter has these dependencies installed.

## load_iris_split(test_size=0.2, random_state=42)

Source: [iris_baseline.py](../iris_baseline.py).

Returns `(X_train, X_test, y_train, y_test, encoder)`. Features are floating
NumPy arrays in this order: SepalLengthCm, SepalWidthCm, PetalLengthCm,
PetalWidthCm. The Id column is excluded. Targets are integer codes; use the
returned encoder to recover Species names.

```python
from iris_baseline import load_iris_split
X_train, X_test, y_train, y_test, encoder = load_iris_split()
print(encoder.inverse_transform(y_test))
```

The default split is stratified and yields 120 training and 30 test rows for
the bundled Iris CSV. Arrays are unscaled: fit preprocessing using training
data only. The CSV path is relative to the module, not the working directory.
Changing random_state changes the split.

## make_classifiers(random_state=42)

Source: [base_models.py](../base_models.py).

Returns a new dictionary of unfitted estimators on every call:

| Key | Estimator configuration |
| --- | --- |
| logistic_regression | LogisticRegression(max_iter=1000) |
| knn | KNeighborsClassifier(n_neighbors=5) |
| svm | SVC(kernel="rbf", random_state=random_state) |
| naive_bayes | GaussianNB() |

The factory order is the table order. The ensemble scripts explicitly select
a different order, beginning with svm. Construct your own dictionary in the
desired training order when calling the ensemble functions.

The factory does not add scaling. Ensemble training functions wrap each model
in a StandardScaler pipeline. The svm factory does not enable probability
estimation; the voting code uses class predictions.

## make_regressors()

Source: [base_models.py](../base_models.py).

Returns fresh, unfitted estimators with keys `linear_regression`,
`knn_regression`, and `svm_regression`. Their configurations are
LinearRegression(), KNeighborsRegressor(n_neighbors=5), and SVR(kernel="rbf").

Choose a numeric target and remove it from the input features. For example,
when predicting PetalLengthCm, use the other three measurements as inputs.
The classifier voting functions are not regression aggregation functions:
do not pass these regressors to predict_bagging or predict_boosting.

## make_kmeans(random_state=42)

Source: [base_models.py](../base_models.py).

Returns a fresh KMeans(n_clusters=3, n_init=10, random_state=random_state).
It is not fitted and does not include a scaler.

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from base_models import make_kmeans
from iris_baseline import load_iris_split

X_train, X_test, _, _, _ = load_iris_split()
clustering = make_pipeline(StandardScaler(), make_kmeans())
clustering.fit(X_train)
cluster_ids = clustering.predict(X_test)
```

Cluster IDs are not the label encoder's Species codes. Comparing these IDs
directly with y_test does not measure classification accuracy.

## train_one_bag(base_model, X_train, y_train, seed)

Source: [bagging.py](../bagging.py).

Returns a fitted Pipeline with steps named `scale` and `model`.
It draws len(X_train) indices with replacement and uses the same indices for
features and targets. Pass NumPy arrays suitable for integer-array indexing.

The estimator is cloned before fitting, so the supplied base_model is not
trained in place. If it exposes random_state, that parameter is replaced by
seed. The scaler and estimator both learn from the sampled bag.

A bootstrap can omit classes, especially for small datasets. This helper does
not retry such bags; estimator fitting errors propagate to its caller.

## fit_bagging(base_models, X_train, y_train, n_jobs=4, seed=42)

Source: [bagging.py](../bagging.py).

Accepts an ordered dictionary of names to estimators and returns a list of
fitted pipelines in dictionary order. There is one learner per entry.
Learner i uses seed + i, starting at zero.

Use n_jobs=1 for sequential execution or -1 to request all available CPUs.
The implementation selects joblib's loky backend and limits inner worker
threads to one. Worker startup is part of the calling script's training timer.

An empty model dictionary, n_jobs=0, or a negative seed raises ValueError.
The names are labels for dictionary entries, not a factory lookup:
pass estimator objects as the values.

## predict_bagging(models, X)

Source: [bagging.py](../bagging.py).

Returns a one-dimensional NumPy array with one voted label per input row.
Each fitted model predicts the same X; the function counts labels per row.
A tied vote selects the first label in NumPy's sorted unique-label order.

```python
from base_models import make_classifiers
from iris_baseline import load_iris_split
from bagging import fit_bagging, predict_bagging

X_train, X_test, y_train, _, encoder = load_iris_split()
models = fit_bagging(make_classifiers(), X_train, y_train, n_jobs=1)
names = encoder.inverse_transform(predict_bagging(models, X_test))
```

An empty models list raises ValueError. Pass raw, unscaled features when using
the returned training pipelines; each pipeline applies its own fitted scaler.
This function returns labels, not probabilities or confidence scores.

## train_one_round(base_model, X_train, y_train, sample_weights, seed)

Source: [boosting.py](../boosting.py).

Returns a fitted scale/model Pipeline around a cloned estimator. Supply one
nonnegative sample weight per training row, normalized to sum to one.

For KNeighborsClassifier, it samples len(X_train) rows with replacement using
sample_weights as sampling probabilities, then fits the pipeline on that bag.
For other estimators, it fits on all training rows and passes
sample_weights * len(X_train) as model__sample_weight.

Custom non-KNN estimators must accept sample_weight in fit. In the direct
weight path, weights are passed to the estimator, not the StandardScaler.
If the estimator exposes random_state, it is set to seed.
