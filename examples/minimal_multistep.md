# minimal_multistep.yaml

Minimal example: one row (eps = 0.4) of Table S5, about 0.5 s.

```bash
igr validate examples/minimal_multistep.yaml
igr run examples/minimal_multistep.yaml --out runs/minimal
igr query runs/minimal summary --json
igr query runs/minimal crossing l1_param 0.05 --json
```

Values obtained (identical to Table S5 from `experiments/fig_s09_s10_table_s5_multistep.py`):

| summary key | value | Table S5 column |
|---|---|---|
| surrogates.param.defect_final | 0.13515837433055547 | delta(theta_1) |
| surrogates.param.l1_final | 0.11699120514655909 | L1 error |
| surrogates.param.defect_integral | 0.1705046997112487 | int_0^1 delta ds |
| gronwall_envelope_eq49 | 27684.960941838923 | Eq(49) envelope |

The smoke test (`tests/test_cli.py::test_smoke_minimal_example`) checks these values.
