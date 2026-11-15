# sweep_eps.yaml

Sweep: the minimal example over the six amplitudes of Table S5, about 1.5 s in total.

```bash
igr validate examples/sweep_eps.yaml           # validates every point, sums the cost estimate
igr sweep examples/sweep_eps.yaml --out runs/sweep_eps
cat runs/sweep_eps/index.json                  # parameters, status, and summary values per point
igr compare runs/sweep_eps/point_000 runs/sweep_eps/point_004 --json
```

Values obtained (surrogates.param.l1_final per point; identical to the L1 error column of Table S5):

| eps | 0.0 | 0.05 | 0.1 | 0.2 | 0.4 | 0.6 |
|---|---|---|---|---|---|---|
| L1 at t=1 | 3.0014e-08 | 0.015915 | 0.031541 | 0.061753 | 0.116991 | 0.163933 |
| defect at t=1 | 1.1230e-05 | 0.019019 | 0.037551 | 0.072889 | 0.135158 | 0.184284 |
