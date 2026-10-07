# Final post-optimization parameter check

Selection uses tuning48 only. B1: four phase/exit neighbors were profiled; top two plus stream/trim controls were warm-checked. A change requires at least2% over the stream baseline in that warm check; this is a noise guard, not a significance test. Unchanged settings reuse their existing full480 paired controls; new settings receive both full480 repeats. SSD controls retain the same seeds and GPU IDs. B8 supersedes that provisional choice with a stability amendment: three warm passes then three measured passes per candidate; select median whole-pass TPS* with a 2% change guard against the original trim control. No slow steps are deleted. The finite neighborhood is not a proof of global optimality.

| Model | B | K1/K2 | exit | stream | DUET AL* | SSD AL* | Full480 ΔAL*95% CI | Selection-excluded432 ΔAL*95% CI | TPS* ratios |
|---|---:|---|---:|---:|---:|---:|---|---|---|
| llama2 | 1 | 11/2 | 21 | 1 | 2.3460 | 2.1425 | [0.16288195601488317, 0.24653554405837602] | [0.15840375609860333, 0.24806158605982032] | 1.088x / 1.133x |
| llama2 | 8 | 3/2 | 21 | 0 | 2.1073 | 2.1508 | [-0.07937074214161778, -0.009417245696974504] | [-0.08109835070559193, -0.009870875610732268] | 0.885x / 0.876x |
| llama3 | 1 | 4/2 | 16 | 1 | 2.6900 | 2.6611 | [-0.01194926054023906, 0.07140904600868718] | [-0.017500200431182687, 0.06945190056188777] | 0.939x / 0.966x |
| llama3 | 8 | 3/1 | 26 | 0 | 2.2843 | 2.6873 | [-0.44141083775300677, -0.3646686002824611] | [-0.44522916054697725, -0.3639998020495575] | 0.786x / 0.781x |
