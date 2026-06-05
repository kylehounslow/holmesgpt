#!/bin/bash
# Shared utilities for in-cluster OpenSearch eval tests. Source at the top of a
# before_test, then:
#   source ../../shared/opensearch_test_utils.sh
#   os_setup "app-300a"
#
# IMPORTANT: setup-time API calls go through `kubectl exec` into the OpenSearch pod
# (curl localhost:9200 inside the container), NOT a port-forward. The harness only
# establishes the test_case.yaml port_forwards later, for the agent's toolset during
# the test itself — they are not available during before_test. (Same approach the
# Loki cases use: kubectl exec ... wget localhost:3100.)
#
# The backend runs with the security plugin disabled (plain HTTP, no auth).
set -uo pipefail

OS_NS=""

# Deploy single-node OpenSearch into a namespace, wait for the pod, then wait for the
# API to report green/yellow health (probed from inside the pod). Usage: os_setup <ns>
os_setup() {
  OS_NS="$1"
  kubectl create namespace "$OS_NS" --dry-run=client -o yaml | kubectl apply -f -
  kubectl apply -f ../../shared/opensearch.yaml -n "$OS_NS"

  echo "⏳ Waiting for OpenSearch pod to be ready (timeout 300s)..."
  local ready=false
  for i in $(seq 1 60); do
    if kubectl wait --for=condition=ready pod -l app=opensearch -n "$OS_NS" --timeout=5s 2>/dev/null; then
      ready=true; break
    fi
    sleep 5
  done
  if [ "$ready" = false ]; then
    echo "❌ OpenSearch pod not ready after 300s"; kubectl get pods -n "$OS_NS" -l app=opensearch; exit 1
  fi

  echo "⏳ Waiting for OpenSearch API health..."
  for i in $(seq 1 60); do
    local status
    status=$(os_req GET "/_cluster/health" 2>/dev/null | grep -oE '"status":"[a-z]+"' | cut -d'"' -f4)
    if [ "$status" = "green" ] || [ "$status" = "yellow" ]; then
      echo "✅ OpenSearch API healthy (status=$status)"; return 0
    fi
    sleep 3
  done
  echo "❌ OpenSearch API not healthy after 180s"; exit 1
}

# Run curl against the OpenSearch REST API from inside the pod (no port-forward needed
# at setup time). Usage: os_req <method> <endpoint> [json_body]
os_req() { # method endpoint [body]
  local method="$1" endpoint="$2" body="${3:-}"
  if [ -n "$body" ]; then
    kubectl exec -i -n "$OS_NS" deployment/opensearch -- \
      curl -sf -X "$method" "http://localhost:9200${endpoint}" \
      -H 'Content-Type: application/json' --data-binary @-  <<<"$body"
  else
    kubectl exec -n "$OS_NS" deployment/opensearch -- \
      curl -sf -X "$method" "http://localhost:9200${endpoint}"
  fi
}

# Create an index with a mapping. Usage: os_create_index <index> <mapping_json>
os_create_index() {
  local index="$1" mapping="$2"
  os_req DELETE "/${index}" >/dev/null 2>&1 || true
  local resp; resp=$(os_req PUT "/${index}?wait_for_active_shards=1" "$mapping")
  echo "$resp" | grep -q '"acknowledged":true' || { echo "❌ index create failed: $resp"; exit 1; }
}

# Bulk-index NDJSON from a file (built with printf to avoid YAML-indentation issues),
# then refresh. Usage: os_bulk <index> <ndjson_file>
os_bulk() {
  local index="$1" file="$2"
  local resp
  resp=$(kubectl exec -i -n "$OS_NS" deployment/opensearch -- \
    curl -sf -X POST "http://localhost:9200/${index}/_bulk" \
    -H 'Content-Type: application/x-ndjson' --data-binary @- < "$file")
  echo "$resp" | grep -q '"errors":false' || { echo "❌ bulk had errors: $resp"; exit 1; }
  os_req POST "/${index}/_refresh" >/dev/null
}

# Append one log doc to an NDJSON bulk file. Usage: os_doc <file> <json_doc_object>
os_doc() { printf '{"index":{}}\n%s\n' "$2" >> "$1"; }

# Assert a _count query returns expected. Usage: os_assert_count <index> <query_json> <expected>
os_assert_count() {
  local index="$1" query="$2" expected="$3"
  local n; n=$(os_req GET "/${index}/_count" "$query" | grep -oE '"count":[0-9]+' | cut -d: -f2)
  [ "$n" = "$expected" ] || { echo "❌ expected count $expected, got ${n:-none}"; exit 1; }
  echo "✅ count check ok ($expected)"
}
