# Async SD sweep — decode_tps (PROFILE_MESA=0, no measurement overhead)

70B AWQ TP=4 + TinyLlama AWQ TP=1, ns=50 in=512 out=512, seed=42, temp=0.7.

## decode_tps (tok/s)

| k \ f |  f=3  |  f=4  |  f=5  |  f=6  |
|---|---:|---:|---:|---:|
| k=7 |  80.35 |  82.99 |  81.07 |  83.65 |
| k=8 |  79.68 |  79.22 |  81.79 |  77.94 |
| k=9 |  78.55 |  80.38 |  71.94 |  73.36 |
| k=10 |  75.32 |  73.93 |  65.75 |  57.66 |