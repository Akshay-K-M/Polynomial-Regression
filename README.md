# Polynomial Regression Assignment

Predicting a continuous target `y` for two datasets (`var1` and `var2`) using only polynomial regression.

- **var1**: 6 input features, true polynomial degree is at most 10
- **var2**: 3 input features, true polynomial degree is at most 20

Each training file has 1000 rows.

## Idea

A polynomial of high degree has a huge number of terms (for example, over 8000 terms for 6 features at degree 10), often more than the number of rows. Plain least squares breaks down in that case. So instead of choosing the degree by hand, the script tries many degrees and several regularised models, and picks the best combination using cross-validation.

## Approach

1. **Features:** for each degree `d`, build all monomials with total degree up to `d`, then standardise each column.
2. **Models**:
   - Ordinary least squares (only when there are fewer terms than rows)
   - Ridge regression (closed form, solved with an SVD so it stays stable)
   - Lasso
   - Elastic net
3. **Selection:** 5-fold cross-validation with the same folds for every model and degree. 
4. **Degree search:** the degree is increased one step at a time and the search stops once the CV error has not improved for 3 degrees in a row. If several degrees are almost tied (within 1%), the lowest one is used.
5. **Final model:** the winning degree, model and penalty are refitted on all training rows and used to predict the test file.

Closed form is used for OLS and ridge instead of gradient descent, since it is exact and avoids tuning a learning rate on badly scaled high-degree features. Lasso and elastic net have no closed form, so they use coordinate descent (scikit-learn).

## Files

| File | Purpose |
|---|---|
| `run.py` | The whole pipeline: features, cross-validation, model selection, plots, predictions |
| `requirements.txt` | Python dependencies |

## Setup

```bash
pip install -r requirements.txt
```

Requirements: `numpy`, `pandas`, `scikit-learn`, `matplotlib`.

## Running

Put the data files in one folder with these names:

```
IMT2024014_train_var1.csv   IMT2024014_test_var1.csv
IMT2024014_train_var2.csv   IMT2024014_test_var2.csv
```

Then run:

```bash
python run.py --data-dir <folder with the csv files> --roll IMT2024014 --out-dir out
```

Options:

| Option | Default | Meaning |
|---|---|---|
| `--data-dir` | `.` | Folder containing the train/test CSV files |
| `--roll` | `IMT2024014` | Roll number used in the file names |
| `--out-dir` | `out` | Where outputs are written |
| `--vars` | `var1 var2` | Which problems to run |

Settings such as the degree ranges, number of folds and penalty grids are constants at the top of `run.py`.

The full run can take a while, since lasso and elastic net on thousands of terms are slow. Most of the time goes to var1 at degrees 6 and above.

## Output

Everything is written to the output folder:

- `IMT2024014_pred_var1.csv`, `IMT2024014_pred_var2.csv`: test predictions, one `y` column
- `cv_results_var1.csv`, `cv_results_var2.csv`: CV MSE and R² for every model and degree
- `cv_curves_var1.png`, `cv_curves_var2.png`: CV MSE and R² against polynomial degree

A summary table and the final chosen model are also printed to the terminal.

## Report

The write up with the reasoning behind these choices is in the report PDF.
