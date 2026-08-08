"""Example datasets: bundled copies first, seaborn for everything else.

This module is a convenience for the tutorials and is not imported by the
core package, so arbital itself has no data dependency.

Four datasets (mpg, penguins, titanic, tips) ship as CSV files inside the
package, so they load offline with no extra installs; these are the
ones the docs and the vignette use.  Any other seaborn dataset name falls
back to seaborn (network on first use), declared as an optional extra::

    pip install "arbital[datasets]"

Any dataset can be loaded by name::

    from arbital import datasets, orbits
    cars = datasets.load("mpg", drop=["name", "origin"])
    orbits(cars, target="mpg").to_html("cars.html")

The convenience wrappers load_mpg(), load_penguins(), load_titanic() and
load_tips() apply sensible column choices for the datasets used in the
docs.

Each loader returns a lightweight Table: a names + float matrix that
orbits() accepts directly.  Categorical columns (string / category /
boolean dtypes, or ones named in `categorical`) are integer-encoded and
recorded in Table.categorical so orbits() picks the correct
mutual-information estimator for them.  Rows with missing values are
dropped, with a note in Table.notes.
"""

from __future__ import annotations

import csv
import os

import numpy as np

__all__ = [
    "Table",
    "available",
    "load",
    "load_mpg",
    "load_penguins",
    "load_tips",
    "load_titanic",
]

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


class Table:
    """A minimal named numeric table (DataFrame-like, no dependency).

    orbits() needs three things from its input: column names (.columns),
    a float matrix (np.asarray), and, optionally, which columns are
    categorical (.categorical).  Table provides exactly those.
    Categorical columns hold integer codes; .levels maps each back to its
    original labels.
    """

    def __init__(self, columns, matrix, categorical=None, levels=None,
                 notes: str = ""):
        self.columns = list(columns)
        self._M = np.asarray(matrix, dtype=float)
        self.categorical = set(categorical or [])
        self.levels = levels or {}
        self.notes = notes

    def __array__(self, dtype=None):
        return self._M if dtype is None else self._M.astype(dtype)

    @property
    def shape(self):
        return self._M.shape

    def column(self, name):
        return self._M[:, self.columns.index(name)]

    def is_categorical(self, name) -> bool:
        return name in self.categorical

    def to_pandas(self):
        """Return a pandas DataFrame with categorical columns decoded."""
        import pandas as pd
        df = pd.DataFrame(self._M, columns=self.columns)
        for name, labels in self.levels.items():
            df[name] = [labels[int(c)] for c in df[name]]
        return df

    def __repr__(self):
        cat = f", categorical={sorted(self.categorical)}" if self.categorical else ""
        return (f"Table({self._M.shape[0]} rows x {len(self.columns)} cols: "
                f"{', '.join(self.columns)}{cat})")


def _seaborn():
    """Import seaborn, with a helpful message if the extra is missing."""
    try:
        import seaborn
    except ImportError as exc:                       # pragma: no cover
        raise ImportError(
            "arbital.datasets requires seaborn; install it with "
            "'pip install \"arbital[datasets]\"'"
        ) from exc
    return seaborn


def _dataframe_to_table(df, drop=None, categorical=None, name="") -> Table:
    """Clean a pandas DataFrame into a Table.

    - `drop`: column names to exclude entirely.
    - `categorical`: column names to force to categorical (in addition to
      any non-numeric dtype, which is detected automatically).
    - rows with any missing value in the kept columns are removed.
    Categorical columns are integer-encoded; the mapping is kept in
    Table.levels.
    """
    drop = set(drop or [])
    forced = set(categorical or [])
    cols = [c for c in df.columns if c not in drop]
    df = df[cols].dropna()

    matrix_cols, cat_cols, levels = [], [], {}
    for c in cols:
        s = df[c]
        is_cat = (c in forced) or (s.dtype.kind in ("O", "b", "U", "S")
                                   or str(s.dtype) == "category")
        if is_cat:
            labels = sorted(map(str, s.unique()))
            code = {lab: i for i, lab in enumerate(labels)}
            matrix_cols.append([float(code[str(v)]) for v in s])
            cat_cols.append(c)
            levels[c] = labels
        else:
            matrix_cols.append([float(v) for v in s])
    M = np.array(matrix_cols).T if matrix_cols else np.empty((0, 0))

    note = f"{name or 'dataset'}: {M.shape[0]} rows"
    if cat_cols:
        note += f"; categorical columns: {', '.join(cat_cols)}"
    return Table(cols, M, cat_cols, levels, note)


def _load_bundled(path, drop=None, categorical=None, name="") -> Table:
    """Read a bundled CSV into a Table with only the standard library.

    Mirrors _dataframe_to_table: a column is categorical when forced by
    `categorical` or when any of its values does not parse as a float;
    rows with a missing value (an empty field, or NaN) are dropped.
    """
    drop = set(drop or [])
    forced = set(categorical or [])
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = [r for r in reader if r]

    keep_idx = [i for i, c in enumerate(header) if c not in drop]
    cols = [header[i] for i in keep_idx]

    missing = {"", "NA", "NaN", "nan"}

    def _as_float(v):
        try:
            f = float(v)
            return None if np.isnan(f) else f
        except ValueError:
            return None

    # column type: numeric unless forced, or unless a non-missing value
    # fails to parse as a float
    values = [[r[i] for r in rows] for i in keep_idx]
    is_cat = [c in forced or any(v not in missing and _as_float(v) is None
                                 for v in col)
              for c, col in zip(cols, values)]

    complete = [all(v not in missing and (cat or _as_float(v) is not None)
                    for v, cat in zip(row_vals, is_cat))
                for row_vals in zip(*values)]
    values = [[v for v, ok in zip(col, complete) if ok] for col in values]

    matrix_cols, cat_cols, levels = [], [], {}
    for c, col, cat in zip(cols, values, is_cat):
        if cat:
            labels = sorted(set(col))
            code = {lab: float(i) for i, lab in enumerate(labels)}
            matrix_cols.append([code[v] for v in col])
            cat_cols.append(c)
            levels[c] = labels
        else:
            matrix_cols.append([float(v) for v in col])
    M = np.array(matrix_cols).T if matrix_cols else np.empty((0, 0))

    note = f"{name or 'dataset'}: {M.shape[0]} rows (bundled copy)"
    if cat_cols:
        note += f"; categorical columns: {', '.join(cat_cols)}"
    return Table(cols, M, cat_cols, levels, note)


def load(name: str, drop=None, categorical=None) -> Table:
    """Load a dataset by name and return a Table.

    Bundled datasets (mpg, penguins, titanic, tips) load from CSV files
    inside the package, offline and dependency-free.  Any other name is
    fetched through seaborn (requires the 'datasets' extra and, on first
    use, a network connection).

    Example::

        datasets.load("diamonds", categorical=["cut", "color", "clarity"])
    """
    bundled = os.path.join(_DATA_DIR, f"{name}.csv")
    if os.path.exists(bundled):
        return _load_bundled(bundled, drop=drop, categorical=categorical,
                             name=name)
    df = _seaborn().load_dataset(name)
    return _dataframe_to_table(df, drop=drop, categorical=categorical, name=name)


def load_mpg() -> Table:
    """Auto MPG: fuel economy against numeric engine and design attributes.

    The free-text 'name' and low-value 'origin' columns are dropped, so
    this is the all-numeric multicollinearity example.
    """
    return load("mpg", drop=["name", "origin"])


def load_penguins() -> Table:
    """Palmer penguins: the four body measurements.

    The species/island/sex columns are dropped so this stays the numeric
    example (bill_depth is confounded by species, a Simpson's-paradox
    setup); use load_titanic() for a categorical example.
    """
    return load("penguins", drop=["species", "island", "sex"])


def load_titanic() -> Table:
    """Titanic: survival against mixed numeric and categorical fields.

    Keeps survived, pclass, sex, age, sibsp, parch, fare, embarked; sex
    and embarked are treated as categorical.  Seaborn's redundant derived
    columns (class, who, deck, ...) are dropped.
    """
    return load("titanic",
                drop=["class", "who", "adult_male", "deck", "embark_town",
                      "alive", "alone"],
                categorical=["sex", "embarked"])


def load_tips() -> Table:
    """Restaurant tips: tip amount against bill, party size, and context.

    total_bill and size are numeric; sex, smoker, day and time are
    categorical.  A compact mixed-type regression example.
    """
    return load("tips")


def available():
    """Names of seaborn's bundled datasets (requires a network connection).

    Any of these can be passed to load().  The convenience wrappers
    (mpg, penguins, titanic, tips) apply tidy defaults and load offline
    from copies bundled with the package.
    """
    return list(_seaborn().get_dataset_names())
