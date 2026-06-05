#!/bin/bash
# Seed a comparison dataset into the SHARED tp-stack OpenSearch (the security-enabled,
# anonymous-via-Dashboards cluster that TestPulsar MCP reads). Used by the MCP-variant
# comparison cases (cmp_*_mcp), which can't seed a per-case security-disabled OpenSearch
# the way the native cases do — TestPulsar targets the shared Dashboards + query-
# enhancements plugin (see memory testpulsar-holmes-integration).
#
# Differs from seed.py (plain-HTTP urllib): tp-stack is HTTPS + auth, and the runner
# reaches it cross-namespace via `kubectl exec` into the OpenSearch pod (admin creds
# read from the pod's own env, so no secret is hardcoded here). The dataset is the SAME
# committed JSON the native arm uses, re-stamped to ~now identically, so the only
# variable remains backend + toolset.
#
# Usage: lvo_seed_tpstack <dataset> <index>
#   <index> should match a Dashboards index pattern TestPulsar discovers (logs-otel-v1*),
#   e.g. logs-otel-v1-bench, so list_index_patterns surfaces it.
set -uo pipefail

LVO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TP_NS="${TP_NS:-tp-stack}"
TP_POD="${TP_POD:-opensearch-cluster-master-0}"

lvo_seed_tpstack() {
  local dataset="$1" index="$2"
  local bulk=/tmp/tp_${dataset}_$$.ndjson

  # Re-stamp the committed dataset to a recent window and build a _bulk body.
  python3 - "$LVO_DIR/datasets/${dataset}.json" "$index" > "$bulk" <<'PY'
import json, sys, datetime
records = json.load(open(sys.argv[1])); index = sys.argv[2]
times = [datetime.datetime.fromisoformat(r["ts"].replace("Z","+00:00")) for r in records]
shift = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=5) - max(times)
for r, t in zip(records, times):
    doc = {k: v for k, v in r.items() if k != "ts"}
    doc["timestamp"] = (t + shift).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    print('{"index":{}}'); print(json.dumps(doc, separators=(",", ":")))
PY

  # Admin password from the OpenSearch pod's own env (chart default for this disposable
  # bench stack). Recreate the index clean, bulk-load over HTTPS, refresh, verify.
  kubectl exec -i -n "$TP_NS" "$TP_POD" -c opensearch -- sh -c '
    P="$OPENSEARCH_INITIAL_ADMIN_PASSWORD"
    curl -sk -u admin:"$P" -X DELETE "https://localhost:9200/'"$index"'" >/dev/null 2>&1 || true
    curl -sk -u admin:"$P" -X POST "https://localhost:9200/'"$index"'/_bulk?refresh=true" \
      -H "Content-Type: application/x-ndjson" --data-binary @- | grep -q "\"errors\":false" \
      && echo "✅ tp-stack bulk ok" || { echo "❌ tp-stack bulk had errors"; exit 1; }
  ' < "$bulk"
  rm -f "$bulk"
}
