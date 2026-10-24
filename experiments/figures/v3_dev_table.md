# v3_dev (dev, 20 worlds x 10 seeds, commit 02d96e33)

| Method | Recall@5 | Recall@10 | Recall@15 | Recall@20 | Recall@25 | Recall@30 | False-robust@10 | Median queries to 1st |
|---|---|---|---|---|---|---|---|---|
| Random | 0.065 | 0.383 | 0.584 | 0.730 | 0.841 | 0.914 | 0.000 | 6.0 |
| Random + pruning | 0.057 | 0.386 | 0.590 | 0.779 | 0.893 | 0.949 | 0.035 | 7.0 |
| CONFUZZ-R v2 | 0.000 | 0.355 | 0.355 | 0.710 | 0.890 | 0.948 | 0.000 | 9.0 |
| Group testing (uniform prior) | 0.000 | 0.355 | 0.355 | 0.710 | 0.890 | 0.941 | 0.000 | 9.0 |

Paired Wilcoxon vs Random + pruning (per world-seed recall difference; descriptive, uncorrected):

- random_minus_random_pruned_recall@10: mean diff -0.003, p=0.82
- random_minus_random_pruned_recall@20: mean diff -0.049, p=0.0099
- v2_minus_random_pruned_recall@10: mean diff -0.031, p=0.0077
- v2_minus_random_pruned_recall@20: mean diff -0.069, p=3.8e-08
- agt_uniform_minus_random_pruned_recall@10: mean diff -0.031, p=0.0077
- agt_uniform_minus_random_pruned_recall@20: mean diff -0.069, p=3.8e-08
