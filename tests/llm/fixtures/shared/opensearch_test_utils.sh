#!/bin/bash
# Shared utilities for in-cluster OpenSearch eval tests. Source at the top of a
# before_test, after deploying shared/opensearch.yaml and starting a port-forward:
#   source ../../shared/opensearch_test_utils.sh
#   os_setup "app-300a" 9301   # namespace, local port for the port-forward
#
# The backend runs with the security plugin disabled (plain HTTP, no auth), so all
# requests are unauthenticated http:// to localhost:$OS_PORT. This mirrors the local
# Loki backend pattern; it is NOT how a production OpenSearch is secured.
set -uo pipefail

OS_PORT=""
OS_URL=""

# Deploy single-node OpenSearch into a namespace and wait for green/yellow health.
# Usage: os_setup <namespace> <local_port>
os_setup() {
  local ns="$1"
  OS_PORT="$2"
  OS_URL="http://localhost:${OS_PORT}"

  kubectl create namespace "$ns" --dry-run=client -o yaml | kubectl apply -f -
  kubectl apply -f ../../shared/opensearch.yaml -n "$ns"

  echo "⏳ Waiting for OpenSearch pod to be ready (timeout 300s)..."
  local ready=false
  for i in $(seq 1 60); do
    if kubectl wait --for=condition=ready pod -l app=opensearch -n "$ns" --timeout=5s 2>/dev/null; then
      echo "✅ OpenSearch pod ready"
      ready=true
      break
    fi
    sleep 5
  done
  if [ "$ready" = false ]; then
    echo "❌ OpenSearch pod not ready after 300s"
    kubectl get pods -n "$ns" -l app=opensearch
    exit 1
  fi
}

# Wait until the port-forwarded API answers and cluster health is green/yellow.
# (The test harness sets up the port-forward from test_case.yaml port_forwards.)
os_wait_api() {
  echo "⏳ Waiting for OpenSearch API at ${OS_URL}..."
  for i in $(seq 1 60); do
    local status
    status=$(curl -sf "${OS_URL}/_cluster/health" 2>/dev/null | grep -oE '"status":"[a-z]+"' | cut -d'"' -f4)
    if [ "$status" = "green" ] || [ "$status" = "yellow" ]; then
      echo "✅ OpenSearch API healthy (status=$status)"
      return 0
    fi
    sleep 3
  done
  echo "❌ OpenSearch API not healthy after 180s"
  exit 1
}

# Create an index with an explicit mapping. Usage: os_create_index <index> <mapping_json>
os_create_index() {
  local index="$1" mapping="$2"
  curl -sf -X DELETE "${OS_URL}/${index}" >/dev/null 2>&1 || true
  local resp
  resp=$(curl -sf -X PUT "${OS_URL}/${index}?wait_for_active_shards=1" \
    -H 'Content-Type: application/json' -d "$mapping")
  echo "$resp" | grep -q '"acknowledged":true' || { echo "❌ index create failed: $resp"; exit 1; }
}

# Bulk-index NDJSON from a file, then refresh. Usage: os_bulk <index> <ndjson_file>
os_bulk() {
  local index="$1" file="$2"
  local resp
  resp=$(curl -sf -X POST "${OS_URL}/${index}/_bulk" \
    -H 'Content-Type: application/x-ndjson' --data-binary @"$file")
  echo "$resp" | grep -q '"errors":false' || { echo "❌ bulk had errors: $resp"; exit 1; }
  curl -sf -X POST "${OS_URL}/${index}/_refresh" >/dev/null
}

# Assert a count query returns an expected number. Usage: os_assert_count <index> <query_json> <expected>
os_assert_count() {
  local index="$1" query="$2" expected="$3"
  local n
  n=$(curl -sf -X GET "${OS_URL}/${index}/_count" -H 'Content-Type: application/json' -d "$query" \
    | grep -oE '"count":[0-9]+' | cut -d: -f2)
  [ "$n" = "$expected" ] || { echo "❌ expected count $expected, got ${n:-none}"; exit 1; }
  echo "✅ count check ok ($expected)"
}
