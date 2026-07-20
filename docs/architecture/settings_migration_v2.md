# Settings Migration V2

P2A defines a non-destructive in-memory migration. It does not overwrite old
settings files.

Target shape:

```json
{
  "settings_schema_version": 2,
  "product": {},
  "profiles": {
    "near": "zenear-v1",
    "blind": "zeblind4d-v1",
    "pipeline": "pipeline-v1"
  }
}
```

Rules:

- `catalog_library_path` migrates to `ProductSettings.catalog_library_path`.
- legacy `db_root`, `index_root`, `blind_4d_manifest_path` are preserved as
  deprecated diagnostics, not deleted.
- historical blind profile is preserved as `historical_diagnostic`, while normal
  product profile becomes `zeblind4d-v1`.
- invalid worker/downsample/timeout values migrate to safe runtime/product
  values.
- internal Near/Blind/dev/benchmark fields are ignored by product settings and
  listed in `SettingsMigrationResult.ignored`.
- no old settings file is destroyed or rewritten by this layer.
