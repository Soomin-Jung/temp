# Deployment and disclosure validation runbook

## 0. Verify assumptions against *actual deployed* source

Reference `source-scrape-values.reference.yaml` came from an unmerged PR, so this must precede deployment.

```bash
# Run ONLY inside HPC administrative network with the required authorization.
curl -fsS 'http://SOURCE_PROMETHEUS:9090/api/v1/status/config'
curl -fsS 'http://SOURCE_PROMETHEUS:9090/api/v1/targets'
# Check actual DCGM modelName label values.
curl -G -fsS 'http://SOURCE_PROMETHEUS:9090/api/v1/query' \
  --data-urlencode 'query=count by (modelName,job) (DCGM_FI_DEV_GPU_UTIL)'
```

Verify actual `job`, `modelName`, `namespace`, `model`, `__name__`, `node`, `instance`, `gpu` label inventory. Verify H200 is identifiable even when a node hosts heterogeneous GPU models. Verify `modelName` for all desired DCGM families; series lacking it fail the H200 filter. Do not assume `model` equals GPU hardware.

## 1. Prove source /federate selection without exposing original server

```bash
curl -G -fsS 'http://SOURCE_PROMETHEUS:9090/federate' \
  --data-urlencode 'match[]={job="nvidia-dcgm-metrics",modelName=~".*H200.*",__name__=~"DCGM_FI_DEV_(GPU_UTIL|FB_USED|FB_FREE|POWER_USAGE|GPU_TEMP)"}'
```

- Confirm positive H200 metric count, metric names, retained labels.
- Confirm no H100/L40S series (label inventory/negative selectors).
- Confirm an external client **cannot access** the original Prometheus or `/federate`.

## 2. Render configuration safely

This Helm example assumes chart `prometheus-community/prometheus` version >= 28 (not kube-prometheus-stack).

```bash
# With a configured internal chart mirror or repository and pinned chart version:
helm show values prometheus-community/prometheus > /tmp/prometheus-chart-defaults.yaml

helm template prometheus-metrics-public prometheus-community/prometheus \
  -n monitoring-public \
  -f federation-h200-only.values.yaml \
  > /tmp/prometheus-public-rendered.yaml

# Inspect rendered ConfigMap prometheus.yml:
# - ONLY the intended federation job is present
# - NO default Kubernetes/node/self-scrape jobs
# - Prometheus service remains ClusterIP
# - no external ingress without auth
# - no Kubernetes API token auto-mount
```

Check Prometheus config using `promtool check config /path/to/rendered/prometheus.yml`; note this input is the **rendered prometheus.yml** extracted from the ConfigMap (NOT the Helm values file). Validate with a pinned Prometheus/promtool image/version in the isolated environment.

## 3. Positive/negative queries on NEW server

```promql
# H200 samples exist:
count(DCGM_FI_DEV_GPU_UTIL{job="nvidia-dcgm-metrics",modelName=~".*H200.*"})

# No other GPU model should be present in the allowed family:
count(DCGM_FI_DEV_GPU_UTIL{job="nvidia-dcgm-metrics",modelName!~".*H200.*"})

# At initial H200-only stage vLLM data must not be present:
count({__name__=~"vllm[:_].*"})

# Check series and label names through HTTP API in addition to PromQL.
```

Test `/api/v1/series`, `/api/v1/labels`, `/api/v1/label/<name>/values`, `/api/v1/targets`, `/api/v1/status/config`; authorized vs unauthorized. The destination creates `up` automatically; metric relabeling cannot delete `up` from the scrape. Check for host/service-name leaks via `instance` and target info. Restrict query APIs with auth/TLS and an API allowlist if needed; new TSDB separation alone does not prevent all metadata disclosure.

## 4. Historic queries and resolution

Let new Federation run > 5 minutes then:

```promql
DCGM_FI_DEV_GPU_UTIL{job="nvidia-dcgm-metrics",modelName=~".*H200.*"}
avg_over_time(DCGM_FI_DEV_GPU_UTIL{job="nvidia-dcgm-metrics",modelName=~".*H200.*"}[5m])
```

Use new server's `/api/v1/query_range` with start before deployment and end after deployment. Expected: only samples ingested since deployment, **no preexisting source history**. After 7 days, 7-day history can be queried subject to retention and any missed scrapes.

Source scrape at 15s + federation at 60s does not copy every intermediate source sample. `rate`, histogram/latency analysis and short bursts may be less precise due to downsampled scrape frequency. If source scrape is 60s, federation at 15s cannot increase source resolution.

## 5. Go-live checklist

- [ ] Manager approves H200 metric name + label allowlist, including server/GPU identification metadata.
- [ ] Source network and Prometheus are not externally reachable.
- [ ] Source job names, H200 `modelName`, scrape intervals and cardinality confirmed.
- [ ] Stage 1 values contains only the H200 job.
- [ ] Client authentication, TLS, firewall/NetworkPolicy, and path-level API policy tested.
- [ ] Negative queries for H100, L40S, vLLM, Kubernetes, LiteLLM, and unapproved DCGM series pass.
- [ ] `up`, targets, config, labels/series endpoints reviewed.
- [ ] PVC capacity and data retention set deliberately.
- [ ] Source and destination CPU/TSDB/cardinality monitored during rollout.
- [ ] vLLM job remains disabled until separately approved; when enabled, validate model/namespace/engine/job scope.
- [ ] Previous source history is NOT promised to client; if needed, design a separate policy-checked backfill/query layer.
