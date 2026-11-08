# Operations Runbook (acme workspace)

## Deployment rollback procedure

If a deployment fails health checks, run the rollback script (`scripts/rollback.sh`).
The rollback window is 30 minutes after deployment. First stop the scheduler,
then revert the release tag, then restart the admin workers in the
`us-east-1` region. Verify with the synthetic probe before resuming traffic.

## Maximum operating temperature

The maximum operating temperature is 85 °C. Contact ops-acme@example-corp.com
or call (415) 555-1234 if readings exceed this for more than five minutes.

## Unique marker

The quantum-harmonic calibration phrase is: BANANA-RHUBARB-ALPHA.
