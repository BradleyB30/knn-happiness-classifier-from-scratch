# K-Nearest Neighbors Happiness Classifier (From Scratch)

CPSC 483 - Introduction to Machine Learning | Programming Assignment 1

Implements a **K-Nearest Neighbors (KNN) classification algorithm from scratch** to predict whether a Fullerton resident is happy or unhappy based on their city satisfaction survey responses. The custom implementation is then benchmarked against **Scikit-learn's KNN**.

---

## Dataset

**Source:** Fullerton, CA Resident Satisfaction Survey (2015-2020)
**File:** `HappinessData-1.csv` — 140 rows, 7 columns

Each row is one resident's survey response. Features are rated on a **1-5 scale**:

| Feature | Description |
|---------|-------------|
| City Services Availability | Quality of city services |
| Housing Cost | Satisfaction with housing affordability |
| Quality of Schools | Rating of local schools |
| Community Trust in Local Police | Trust in local law enforcement |
| Community Maintenance | Upkeep of public spaces |
| Availability of Community Room | Access to community facilities |

**Target variable:** `Unhappy/Happy` — `0` = Unhappy, `1` = Happy

> Note: Some survey entries are blank (question not asked that year) or NA (no response). These are handled during preprocessing.

---

## What the Code Does

### Step-by-step workflow

1. **Load & restructure** — Reads the CSV and moves the class label from the first column to the last column (ML convention)
2. **Preprocess** — Fills missing/blank values with the column mode (most common value)
3. **Pearson Correlation** — Measures how strongly each feature relates to happiness
4. **KNN from scratch** — Implements the algorithm using only Python's standard library, tested with Euclidean, Manhattan, and Minkowski distances
5. **Train/Test Split** — 80% training, 20% testing (112 train, 28 test)
6. **Scikit-learn comparison** — Runs sklearn's KNN on the same split for comparison
7. **Elbow plot** — Tests k=1 through k=40 and plots error rate to find the optimal k
8. **5-Fold Cross-Validation** — From scratch; rotates train/test splits for a more reliable accuracy estimate
9. **Metrics** — Confusion matrix, accuracy, precision, recall, F1 score, and specificity

---

## Key Results

| Model | k | Accuracy | F1 Score |
|-------|---|----------|----------|
| Custom KNN (Euclidean) | 10 | 0.7857 | 0.8333 |
| Custom KNN (Manhattan) | 5 | 0.6786 | 0.7429 |
| Scikit-learn KNN | 5 | 0.6786 | — |

**Best k:** 10 (selected via elbow plot)
**Strongest feature correlation with happiness:** City Services Availability (r = +0.32)

---

## Files

```
CPSC483PA1.py                      # Runnable Python script (all 10 tasks)
CPSC483PA1.ipynb                   # Jupyter notebook (same code + explanations)
HappinessData-1.csv                # Original dataset
HappinessData-1-restructured.csv  # Auto-generated: label moved to last column
elbow_plot.png                     # Error Rate vs. K chart
confusion_matrix.png               # Confusion matrix side-by-side
```

---

## How to Run

**Requirements:**
```bash
pip install scikit-learn matplotlib
```

**Run the Python script:**
```bash
python CPSC483PA1.py
```

**Run the Jupyter notebook:**
```bash
jupyter notebook CPSC483PA1.ipynb
```
Or open `CPSC483PA1.ipynb` directly in VS Code and click **Run All**.

---

## Libraries Used

| Library | Purpose |
|---------|---------|
| `csv`, `math`, `random`, `collections` | Python standard library — used for all custom implementations |
| `matplotlib` | Plots (elbow chart, confusion matrix, correlation heatmap) |
| `scikit-learn` | KNN comparison model only (Task 6) |

> The KNN algorithm, train/test split, cross-validation, Pearson correlation, and all metrics are implemented from scratch using only the Python standard library.

---

## Distance Metrics

Three metrics were tested and compared at k=5:

- **Euclidean (L2)** — straight-line distance between two points
- **Manhattan (L1)** — sum of absolute differences per feature
- **Minkowski (p=3)** — generalization of the above

All three produced equal accuracy on this dataset at k=5. Euclidean was selected as the primary metric.
