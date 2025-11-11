# zeblindsolver

`zeblindsolver` is the pure-Python blind solver suite for ZeSolver. It expects ASTAP/HNSKY catalogs laid out under `database/` and provides two CLIs:

## Build an index

1. Generate a tangent-plane tile cache from your ASTAP databases:

```bash
zebuildindex --db-root /path/to/database --index-root /path/to/index --mag-cap 15.5 --max-stars 2000
```

This converts every ASTAP tile (`.1476`, `.290`) into `(x_deg,y_deg,ra_deg,dec_deg,mag)` `.npz` blobs, writes a `manifest.json`, and then builds the multi-scale quad hashes for levels `L`, `M`, and `S` under `hash_tables/`. Adjust `--max-quads-per-tile` if you want to limit how many quads are hashed per tile (default `20000`).

## Run the blind solver

```bash
zeblindsolve input.fits --index-root /path/to/index \
    --max-candidates 12 --max-stars 800 --max-quads 12000 \
    --sip-order 2 --quality-rms 1.0 --quality-inliers 60 --log-level INFO
```

The solver detects stars, builds observed quads, looks up hashed tiles, estimates a similarity transform, and writes back a TAN WCS with quality metrics. Successful solves set `SOLVED=1`, `SOLVER=ZeSolver`, `SOLVMODE=BLIND`, `BLINDVER`, `DBSET`, `TILE_ID`, `RMSPX`, `INLIERS`, `SIPORD`, and `QUALITY`.

GUI users can enable "Fast mode (S-only, fallback M/L)" from the settings to try the most selective level first and improve speed.

## Limitations

- The current matcher is tuned for fields with 400–1000 stars and assumes ASTAP-style tiles. Very sparse/very dense frames may require tuning.
