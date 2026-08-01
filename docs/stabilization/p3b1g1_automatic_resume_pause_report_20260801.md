# P3B-1G1 - Automatic Resume, Pause and Partial Download Preservation

## State Machine

P3B-1G1 adds a GUI-free `DistributionTransferController` used by the distribution engine and the startup wizard worker. Its explicit states are:

- `idle`
- `downloading`
- `retry_wait`
- `paused`
- `cancelling`
- `completed`
- `failed`

Transitions are centralized in the controller through `transition`, `request_pause`, `request_resume`, `request_resume_now`, `request_cancel`, `wait_for_retry`, and `wait_if_paused`.

## Retry Policy

Recoverable component failures no longer terminate the full distribution run immediately. A failing component now tries the enabled sources in order, then enters an interruptible retry wait if all active sources fail temporarily.

Default retry delays are:

- first interruption: 10 seconds
- second interruption: 30 seconds
- third and later interruptions: 60 seconds

The delay is capped at 60 seconds. The production policy is bounded by `max_component_retries=8` to avoid an infinite unattended run, while tests can lower the delay and retry count.

Recoverable failures include network unavailability, timeouts, connection reset, `WinError 10054`, temporary DNS/network text, HTTP 408/429/500/502/503/504, and incomplete files smaller than the expected size.

Non-recoverable failures include invalid ranges, SHA-256 mismatch, oversized partials, archive/package/manifest/storage/destination failures, and final library validation failures.

## Resume Now

During `retry_wait`, `request_resume_now` wakes the controller event immediately. The component resumes without waiting for the timer, retries enabled sources, and starts from the current `.part` size when the partial is compatible.

Manual retry failure returns cleanly to `retry_wait` with the next retry number and delay.

## Pause Versus Network Error

Network error:

- schedules automatic retry
- emits `retry_wait`
- keeps Cancel active
- allows immediate `Reprendre maintenant`
- may continue other healthy component downloads in parallel

User pause:

- requests a cooperative stop
- preserves `.part` files
- emits `paused`
- disables automatic retry until explicit resume
- resumes incomplete components from cache/partials

## Partial Files

Rules implemented:

- valid final cache file: reused, no network request
- `.part` smaller than expected after temporary failure: preserved for retry
- `.part` larger than expected: discarded for that component
- complete file with invalid SHA-256: isolated/rejected
- server ignores Range with `200 OK`: restarts only that component
- inconsistent `Content-Range`: rejects the component and discards that partial

When a partial from one source ends with a size mismatch and another source is available, the partial is discarded before source fallback to avoid appending bytes from a different payload.

## Multi-Source and Parallelism

Enabled sources remain ordered by policy. Dormant mirrors are data-only and still produce no requests.

Temporary failure is scoped to the component. Other component futures keep running. Completed components remain in cache and are reused. The resume path restores normal component concurrency under `max_parallel_downloads`.

## Cancellation and Close

Cancel sets `cancelling`, wakes retry waits and pauses, prevents new retries, and preserves `.part` files. The wizard now asks explicitly when closing during an active, paused, or retrying download:

- `Quitter et conserver`
- `Continuer le telechargement`
- `Annuler l'installation`

No network operation is intentionally left running silently after wizard close.

## Telemetry

Added structured events:

- `DISTRIBUTION_RETRY_SCHEDULED`
- `DISTRIBUTION_RETRY_BEGIN`
- `DISTRIBUTION_RETRY_NOW_REQUESTED`
- `DISTRIBUTION_RETRY_CANCELLED`
- `DISTRIBUTION_PAUSE_REQUESTED`
- `DISTRIBUTION_PAUSED`
- `DISTRIBUTION_RESUMED`

`DISTRIBUTION_RUN_END` now carries non-zero download stats after cancellation or failure when bytes were actually processed:

- `bytes_downloaded`
- `bytes_resumed`
- `bytes_reused`
- `max_concurrency_observed`
- `sources_used`
- `retry_count`
- `pause_count`
- `status`
- `total_duration_s`

## Automated Tests

Validated:

- timeout then automatic retry
- second timeout with second retry delay
- retry delay cap
- `WinError 10054` classification through retry
- resume-now during retry wait
- resume-now failure returning to retry wait
- pause during retry wait then resume
- cancel during retry wait
- two interrupted components resuming independently
- completed component not redownloaded
- valid cache with no network request
- server ignoring Range
- inconsistent size and SHA handling
- source fallback
- dormant mirrors not requested
- no duplicate concurrent asset download
- wizard pause/resume/retry labels and close confirmation source audit

Command packet:

```text
.venv/bin/python -m pytest tests/test_catalog_distribution_multisource.py tests/test_catalog_distribution.py tests/test_startup_wizard.py -q
82 passed
```

## Manual Windows Test

Not executed in this Linux development session. Required Windows validation:

1. empty cache
2. start GitHub Releases download
3. interrupt Wi-Fi or VPN
4. confirm automatic `retry_wait`
5. restore network
6. click `Reprendre maintenant`
7. confirm immediate `.part` resume
8. pause voluntarily
9. resume
10. finish installation
11. confirm `READY_FULL`

Expected log evidence:

- immediate retry after click
- no forced timer wait
- `.part` preservation
- no redownload of completed components
- restored parallelism
- valid SHA-256
- atomic installation success

## Remaining Limits

- The production retry policy is bounded to avoid an unattended infinite run; a user can still restart later and reuse valid cache/partials.
- Pause is cooperative: an active blocking socket read exits when the HTTP backend raises, times out, returns data, or the underlying request notices cancellation.
- Windows manual network interruption remains the final acceptance proof.
