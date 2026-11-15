# typical_s1_small_sinusoid.yaml

Typical example: Experiment S1 (Figures S1, S2), "quadratic + small sinusoid", about 1 s.

```bash
igr run examples/typical_s1_small_sinusoid.yaml --out runs/typical
igr query runs/typical summary --json
igr query runs/typical final --json          # density moments and modes at t = 2
igr query runs/typical series l1_param --points 20 --json
```

Values obtained (the S1 script prints the same numbers to 4 significant figures:
L1 param 0.1291, L1 proj 0.08293, KL param 0.0129, KL proj 0.01184):

| summary key | value |
|---|---|
| surrogates.param.l1_final | 0.12906065174751657 |
| surrogates.proj.l1_final | 0.08292681902465551 |
| surrogates.param.kl_final | 0.012901094721042013 |
| surrogates.proj.kl_final | 0.011839356720529543 |
| true.mu_final | 0.5385261155954715 |
| surrogates.param.mu_final | 0.5119464893454682 |
