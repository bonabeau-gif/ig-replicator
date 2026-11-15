# projection_linear_shaping.yaml

Non-default model: Table S1 (linear fitness under rank shaping), about 0.6 s.

```bash
igr run examples/projection_linear_shaping.yaml --out runs/table_s1
igr query runs/table_s1 table --where dictionary=gaussian --columns target,r2 --json
igr query runs/table_s1 table --where "target=elite tau=0.25" --columns dictionary,r2 --json
```

Values obtained (R^2; identical to Table S1 from `experiments/fig_s05_table_s1_linear_shaping.py`):

| dictionary | linear | elite tau=0.25 | smooth_elite tau=0.25 h=0.05 |
|---|---|---|---|
| gaussian + legendre_rank K=16 | 1.000000 | 0.956279 | 0.999996 |
| gaussian + legendre_rank K=2 | 1.000000 | 0.818494 | 0.959075 |
| gaussian + legendre_rank K=4 | 1.000000 | 0.855990 | 0.981359 |
| gaussian + legendre_rank K=8 | 1.000000 | 0.925178 | 0.999334 |
| gaussian + rank_bins K=8 | 0.993657 | 1.000000 | 0.974544 |
| gaussian | 0.954930 | 0.661078 | 0.809607 |

The Gaussian/elite entry 0.661078 is Eq. (S4), R^2 = 0.661.
