# =============================================================
# CPSC 483 x Programming Assignment 1
# KNN Happiness Classifier from Scratch
# =============================================================

import csv
import math
import random
from collections import Counter

# ------------------------------------------------------------x
# TASK 1 & 2: Load CSV and move class label to last column
# ------------------------------------------------------------x
def load_and_restructure(filename):
    """Load CSV, move first column (class label) to last, save result."""
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)

    header    = rows[0]
    data_rows = rows[1:]

    # Move first column to last position
    new_header = header[1:] + [header[0]]
    new_data   = [row[1:] + [row[0]] for row in data_rows]

    # Save restructured CSV
    out_file = 'HappinessData-1-restructured.csv'
    with open(out_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(new_header)
        writer.writerows(new_data)

    print("Original header    :", header)
    print("Restructured header:", new_header)
    print(f"Restructured CSV saved -> {out_file}\n")
    return new_header, new_data


FILENAME = 'HappinessData-1.csv'
header, raw_data = load_and_restructure(FILENAME)
print(f"Dataset loaded: {len(raw_data)} rows x {len(header)} columns")


# ------------------------------------------------------------x
# TASK 3: Preprocessing x Handle Missing / NA Values
# ------------------------------------------------------------x
NUM_FEATURES = 6   # columns 0-5 are features; column 6 is the label


def column_mode(data, col_idx):
    """Return the mode of a feature column, ignoring blanks and NA."""
    vals = []
    for row in data:
        v = row[col_idx].strip()
        if v and v.upper() != 'NA':
            vals.append(float(v))
    return Counter(vals).most_common(1)[0][0]


def preprocess(raw_data):
    """Fill missing values with column mode and convert to numeric."""
    modes = [column_mode(raw_data, i) for i in range(NUM_FEATURES)]

    print("-- Missing-value imputation (mode per column) --")
    for i, col_name in enumerate(header[:NUM_FEATURES]):
        n_missing = sum(
            1 for r in raw_data
            if r[i].strip() == '' or r[i].strip().upper() == 'NA'
        )
        print(f"  {col_name:<42s}  mode={modes[i]}  missing={n_missing}")

    processed = []
    for row in raw_data:
        new_row = []
        for i in range(NUM_FEATURES):
            v = row[i].strip()
            new_row.append(modes[i] if (v == '' or v.upper() == 'NA') else float(v))
        new_row.append(int(row[NUM_FEATURES].strip()))
        processed.append(new_row)

    total_missing = sum(
        1 for row in raw_data
        for i in range(NUM_FEATURES)
        if row[i].strip() == '' or row[i].strip().upper() == 'NA'
    )
    print(f"\nTotal missing values imputed: {total_missing}\n")
    return processed


data = preprocess(raw_data)


# ------------------------------------------------------------x
# TASK 4: Pearson Correlation (from scratch)
# ------------------------------------------------------------x
def pearson(x, y):
    """Compute Pearson correlation coefficient between two lists."""
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    numerator = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    dx = math.sqrt(sum((v - mx) ** 2 for v in x))
    dy = math.sqrt(sum((v - my) ** 2 for v in y))
    return numerator / (dx * dy) if dx and dy else 0.0


columns    = [[row[i] for row in data] for i in range(NUM_FEATURES + 1)]
target_col = columns[NUM_FEATURES]

print("-- Pearson Correlation: each feature vs. target (Happiness) --")
for i in range(NUM_FEATURES):
    print(f"  {header[i]:<42s}: {pearson(columns[i], target_col):+.4f}")

print("\n-- FeaturexFeature Correlation Matrix --")
short_names = [h[:14] for h in header[:NUM_FEATURES]]
print(f"{'':16}", end='')
for s in short_names:
    print(f"{s:>15}", end='')
print()
for i in range(NUM_FEATURES):
    print(f"{short_names[i]:16}", end='')
    for j in range(NUM_FEATURES):
        print(f"{pearson(columns[i], columns[j]):>15.3f}", end='')
    print()
print()


# ------------------------------------------------------------x
# TASK 5: KNN from Scratch x Distance Metrics
# ------------------------------------------------------------x
def euclidean(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a))))


def manhattan(a, b):
    return sum(abs(a[i] - b[i]) for i in range(len(a)))


def minkowski(a, b, p=3):
    return sum(abs(a[i] - b[i]) ** p for i in range(len(a))) ** (1.0 / p)


def knn_predict(train_X, train_y, test_X, k, dist_fn):
    """Predict labels for all test points using KNN."""
    preds = []
    for point in test_X:
        neighbors = sorted(
            ((dist_fn(point, train_X[i]), train_y[i]) for i in range(len(train_X))),
            key=lambda x: x[0]
        )
        votes = [neighbors[j][1] for j in range(k)]
        preds.append(Counter(votes).most_common(1)[0][0])
    return preds


# ------------------------------------------------------------x
# Train / Test Split x 80 / 20 (from scratch)
# ------------------------------------------------------------x
def split_data(data, test_ratio=0.2, seed=42):
    random.seed(seed)
    d = list(data)
    random.shuffle(d)
    cut = int(len(d) * (1 - test_ratio))
    return d[:cut], d[cut:]


train_data, test_data = split_data(data)
train_X = [r[:-1] for r in train_data]
train_y = [r[-1]  for r in train_data]
test_X  = [r[:-1] for r in test_data]
test_y  = [r[-1]  for r in test_data]
print(f"Train size: {len(train_X)}   Test size: {len(test_X)}\n")


def accuracy(y_true, y_pred):
    return sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)


# Compare distance metrics at k = 5
print("-- Custom KNN  k=5: Distance Metric Comparison --")
METRICS = [
    ("Euclidean",       euclidean),
    ("Manhattan",       manhattan),
    ("Minkowski(p=3)",  lambda a, b: minkowski(a, b, 3)),
]
metric_results = {}
for name, fn in METRICS:
    preds = knn_predict(train_X, train_y, test_X, 5, fn)
    acc   = accuracy(test_y, preds)
    metric_results[name] = acc
    print(f"  {name:<20s}  Accuracy={acc:.4f}  Error={1-acc:.4f}")

best_metric_name = max(metric_results, key=metric_results.get)
best_metric_fn   = dict(METRICS)[best_metric_name]
print(f"\nBest distance metric: {best_metric_name}\n")


# ------------------------------------------------------------x
# TASK 6: Scikit-learn KNN Comparison  (k = 5)
# ------------------------------------------------------------x
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                              ConfusionMatrixDisplay)

sk5 = KNeighborsClassifier(n_neighbors=5)
sk5.fit(train_X, train_y)
sk5_preds = list(sk5.predict(test_X))
sk5_acc   = accuracy_score(test_y, sk5_preds)

print("-- Scikit-learn KNN  k=5 --")
print(f"  Accuracy={sk5_acc:.4f}   Error={1-sk5_acc:.4f}\n")


# ------------------------------------------------------------x
# TASK 7 & 8: Find Best k x Elbow Plot (Error Rate vs. k)
# ------------------------------------------------------------x
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for .py script
import matplotlib.pyplot as plt

K_RANGE        = range(1, 41)
custom_errors  = []
sklearn_errors = []

print("Computing error rates for k = 1 x 40 x")
for k in K_RANGE:
    p_custom = knn_predict(train_X, train_y, test_X, k, euclidean)
    custom_errors.append(1 - accuracy(test_y, p_custom))

    sk = KNeighborsClassifier(n_neighbors=k)
    sk.fit(train_X, train_y)
    sklearn_errors.append(1 - accuracy_score(test_y, sk.predict(test_X)))

# -- Elbow plot --
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, errors, title in zip(
        axes,
        [custom_errors, sklearn_errors],
        ['Custom KNN (Euclidean)', 'Scikit-learn KNN']):
    ax.plot(list(K_RANGE), errors, 'b--o',
            markerfacecolor='red', markersize=6, linewidth=1.2)
    ax.set_title(f'Error Rate vs. K Value\n({title})', fontsize=12)
    ax.set_xlabel('k value')
    ax.set_ylabel('Error Rate')
    ax.set_xticks(range(0, 41, 5))
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('elbow_plot.png', dpi=120, bbox_inches='tight')
print("Elbow plot saved -> elbow_plot.png")

best_k_custom  = list(K_RANGE)[custom_errors.index(min(custom_errors))]
best_k_sklearn = list(K_RANGE)[sklearn_errors.index(min(sklearn_errors))]
print(f"\nBest k (Custom KNN)  : {best_k_custom}  "
      f"(Error={min(custom_errors):.4f})")
print(f"Best k (Scikit-learn): {best_k_sklearn}  "
      f"(Error={min(sklearn_errors):.4f})\n")


# ------------------------------------------------------------x
# TASK 9: N-Fold Cross-Validation (from scratch)
# ------------------------------------------------------------x
def cross_validate(data, n_folds, k, dist_fn, seed=42):
    """Perform n-fold cross-validation and return (mean_acc, std, fold_accs)."""
    random.seed(seed)
    d = list(data)
    random.shuffle(d)

    fold_sz = len(d) // n_folds
    folds   = [d[i * fold_sz:(i + 1) * fold_sz] for i in range(n_folds)]
    # distribute remainder rows
    for idx, row in enumerate(d[n_folds * fold_sz:]):
        folds[idx % n_folds].append(row)

    fold_accs = []
    for i in range(n_folds):
        test_fold  = folds[i]
        train_fold = [r for j, fold in enumerate(folds) if j != i for r in fold]
        tX = [r[:-1] for r in train_fold];  ty = [r[-1] for r in train_fold]
        eX = [r[:-1] for r in test_fold];   ey = [r[-1] for r in test_fold]
        preds = knn_predict(tX, ty, eX, k, dist_fn)
        fold_accs.append(accuracy(ey, preds))

    mean = sum(fold_accs) / len(fold_accs)
    std  = math.sqrt(sum((a - mean) ** 2 for a in fold_accs) / len(fold_accs))
    return mean, std, fold_accs


print("-- 5-Fold Cross-Validation  (Custom KNN, Euclidean) --")
print(f"{'k':>4}  {'Mean Acc':>10}  {'Std Dev':>8}  Fold Accuracies")
print("-" * 70)
for k in sorted({3, 5, 7, best_k_custom}):
    mean, std, fold_accs = cross_validate(data, 5, k, euclidean)
    folds_str = '  '.join(f'{a:.3f}' for a in fold_accs)
    print(f"{k:>4}  {mean:>10.4f}  {std:>8.4f}  [{folds_str}]")
print()


# ------------------------------------------------------------x
# TASK 10: Confusion Matrix & Full Classification Metrics
# ------------------------------------------------------------x
def binary_metrics(y_true, y_pred, label=''):
    """Print confusion matrix, accuracy, precision, recall, F1, specificity."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)

    total = tp + fp + fn + tn
    acc   = (tp + tn) / total       if total        else 0.0
    prec  = tp / (tp + fp)          if (tp + fp)    else 0.0
    rec   = tp / (tp + fn)          if (tp + fn)    else 0.0
    f1    = 2*prec*rec/(prec+rec)   if (prec + rec) else 0.0
    spec  = tn / (tn + fp)          if (tn + fp)    else 0.0

    bar = '=' * 54
    print(f"\n{bar}")
    print(f"  {label}")
    print(bar)
    print(f"  Confusion Matrix (rows=Actual, cols=Predicted):")
    print(f"  {'':<20}  {'Pred 0':>8}  {'Pred 1':>8}")
    print(f"  {'Actual 0 (Unhappy)':<20}  {tn:>8}  {fp:>8}")
    print(f"  {'Actual 1 (Happy)':<20}  {fn:>8}  {tp:>8}")
    print(f"\n  Accuracy    : {acc:.4f}")
    print(f"  Precision   : {prec:.4f}")
    print(f"  Recall      : {rec:.4f}")
    print(f"  F1 Score    : {f1:.4f}")
    print(f"  Specificity : {spec:.4f}")
    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1, specificity=spec)


# Custom KNN x best k (Euclidean)
best_preds_custom = knn_predict(train_X, train_y, test_X, best_k_custom, euclidean)
binary_metrics(test_y, best_preds_custom,
               f'Custom KNN  |  k={best_k_custom}  |  Euclidean  |  Test Set')

# Custom KNN x k=5 Manhattan
preds_manhattan = knn_predict(train_X, train_y, test_X, 5, manhattan)
binary_metrics(test_y, preds_manhattan,
               'Custom KNN  |  k=5  |  Manhattan  |  Test Set')

# Scikit-learn x best k
sk_best = KNeighborsClassifier(n_neighbors=best_k_sklearn)
sk_best.fit(train_X, train_y)
sk_best_preds = list(sk_best.predict(test_X))
binary_metrics(test_y, sk_best_preds,
               f'Scikit-learn KNN  |  k={best_k_sklearn}  |  Test Set')

# Full Scikit-learn report (k=5)
print("\n-- Scikit-learn Classification Report  (k=5) --")
print(classification_report(test_y, sk5_preds,
                             target_names=['Unhappy (0)', 'Happy (1)']))

# Confusion matrix visualisation
fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4))
for ax, preds, title in zip(
        axes2,
        [best_preds_custom, sk_best_preds],
        [f'Custom KNN (k={best_k_custom})',
         f'Scikit-learn KNN (k={best_k_sklearn})']):
    ConfusionMatrixDisplay.from_predictions(
        test_y, preds,
        display_labels=['Unhappy', 'Happy'],
        cmap='Blues', ax=ax
    )
    ax.set_title(title)

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=120, bbox_inches='tight')
print("Confusion matrix plot saved -> confusion_matrix.png")
