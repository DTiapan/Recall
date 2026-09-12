# Cloud Infrastructure Runbook — Production Retrieval Tier

**Service:** recall-retrieval-prod  
**Last Updated:** 2026-08-15  
**On-call:** platform-infra@recall.io

## Overview

This runbook covers the hybrid retrieval tier (Qdrant dense + sparse vectors) serving `/v1/search` in the us-east-1 production cluster.

## SLOs

- **P99 retrieval latency**: ≤ 150 ms under normal load (10K QPS burst)
- Availability: 99.95% monthly
- Error budget: 0.05% (≈ 22 minutes downtime/month)

## Circuit Breaker Behavior

When downstream Qdrant shard latency exceeds 200 ms for 3 consecutive probe windows, the **circuit breaker** opens and the API **degrades to sparse-only retrieval** for 60 seconds. Clients receive `X-Degraded-Mode: sparse` response headers.

## Kubernetes Failover — Pod Eviction

During node maintenance or memory pressure, the cluster autoscaler may trigger **Kubernetes pod eviction**. The retrieval Deployment uses PodDisruptionBudget `minAvailable: 2` across three zones. If eviction drains a zone:

1. Verify HPA has scaled replicas ≥ 4 (`kubectl get hpa recall-retrieval`)
2. Confirm Qdrant cluster quorum (green status in Grafana dashboard `retrieval/qdrant-health`)
3. Drain is safe when all pods report `Ready` on surviving nodes

## Rollback Procedure

1. `kubectl rollout undo deployment/recall-retrieval -n prod`
2. Validate canary error rate < 0.1% for 5 minutes
3. Post incident summary to `#platform-incidents`

## Escalation

| Severity | Condition | Action |
|----------|-----------|--------|
| SEV1 | P99 > 500 ms for 10 min | Page on-call + VP Engineering |
| SEV2 | Circuit breaker open > 15 min | On-call investigates shard rebalance |
| SEV3 | Single pod eviction | Auto-remediate; no page |
