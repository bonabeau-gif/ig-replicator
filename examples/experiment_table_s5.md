# experiment_table_s5.yaml

Experiment model: runs `experiments/fig_s09_s10_table_s5_multistep.py` unchanged, about 2 s.
Tables and figures land in the run directory, not in the repository's `tables/` and `figures/`.

```bash
igr run examples/experiment_table_s5.yaml --out runs/exp_s09
igr query runs/exp_s09 table --file tableS5_multistep --json
```

Values obtained: `outputs/tables/tableS5_multistep.csv` and both figures are byte-identical to
the files `python run_all.py S09` writes. The eps = 0.4 row is
0.13515837433055547, 0.11699120514655909, 0.1705046997112487, 27684.960941838923
(the same numbers as `minimal_multistep.yaml`).
