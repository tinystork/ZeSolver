# ZN3.10B GUI fallback 4D manual test

Dataset: `/home/tristan/near_bench_cmp30/zn310b_gui_fallback4d_20260716_153931`
GUI batch directory: `/home/tristan/near_bench_cmp30/zn310b_gui_fallback4d_20260716_153931/gui_mixed`
Manifest: `/home/tristan/.openclaw/workspace/projects/ZeSolver/reports/zenear_zn310b_gui_manifest.json`

1. Launch the GUI from this project Python environment.
2. Select `/home/tristan/near_bench_cmp30/zn310b_gui_fallback4d_20260716_153931/gui_mixed` as the input directory.
3. Use the local solver backend.
4. Verify the 4D manifest is valid: `/home/tristan/.openclaw/workspace/projects/ZeSolver/config/zeblind_4d_experimental_manifest.json`.
5. Verify six 4D indexes are loaded.
6. Disable Astrometry.net web for this run: no web backend, no API fallback.
7. Keep `strict_acceptance_mode = diagnostic`.
8. Enable WCS writing/overwrite for these test copies.
9. Before launch, confirm the GUI/log shows:
   - `Chaîne effective : ZeNear → ZeBlind 4D`
   - `Strict acceptance mode: diagnostic`
   - `Astrometry.net web: disabled`
10. Run the batch once.
11. Keep the complete log under `/home/tristan/near_bench_cmp30/zn310b_gui_fallback4d_20260716_153931/logs`.
12. Do not rerun the same lot before saving the result files and log.

After the run, analyze it with:

```bash
python tools/diagnose_zn310b_gui_fallback.py \
  --log /home/tristan/near_bench_cmp30/zn310b_gui_fallback4d_20260716_153931/logs/<gui-log-file>.log \
  --manifest reports/zenear_zn310b_gui_manifest.json \
  --oracle-root /home/tristan/near_bench_cmp30/zn310b_gui_fallback4d_20260716_153931/oracle_sidecars \
  --output reports/zenear_zn310b_gui_result.json
```
