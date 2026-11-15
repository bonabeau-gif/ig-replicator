# projection_multimodal_rbf.yaml

Non-default dictionaries: four rows of Table S3 (multimodal fitness, elite top-25%), about 0.5 s.

```bash
igr run examples/projection_multimodal_rbf.yaml --out runs/table_s3
igr query runs/table_s3 table --columns dictionary,r2,cond --json
```

Values obtained (identical to the corresponding rows of Table S3 from
`experiments/fig_s07_table_s3_multimodal.py`):

| dictionary | R^2 | Gram condition number |
|---|---|---|
| gaussian + rank_bins K=8 | 1.000000 | 8.9e16 (rank bins are exact by construction, ill-conditioned) |
| gaussian + legendre_rank K=8 | 0.923384 | 65.24 |
| gaussian + rbf K=9 h=0.8 | 0.818588 | 1723.8 |
| gaussian | 0.022218 | 1.984 |
