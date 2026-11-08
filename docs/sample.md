# Engineering Handbook

## Deployment Rollback

If a production deployment fails health checks, the standard rollback procedure is:

1. Run `kubectl rollout undo deployment/<name>` in the target namespace.
2. Confirm the previous ReplicaSet is at full readiness via `kubectl rollout status`.
3. Notify `#oncall` in Slack with the failure reason and rollback timestamp.

## Operating Temperature

The firmware module supports a maximum continuous operating temperature of 85 °C.
Extended operation above 85 °C voids the warranty and can cause thermal shutdown.