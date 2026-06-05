#!/bin/bash
# Shared helpers for the Loki-vs-OpenSearch comparison cases (cmp_*). Each scenario is
# a matched pair cmp_<scenario>_loki / cmp_<scenario>_opensearch that seeds the SAME
# committed dataset (shared/lvo/datasets/<scenario>.json) into its backend, so the only
# variable is the backend + its HolmesGPT toolset.
#
# Seeding runs from the test-runner pod over CLUSTER DNS (<svc>.<ns>.svc.cluster.local),
# not a port-forward: before_test executes in the runner pod, which is in-cluster, so it
# reaches the backend Service directly. (The harness only wires the test_case.yaml
# port_forwards for the AGENT's toolset during the test, not during before_test.) The
# agent path still uses localhost:<port> via those port_forwards.
#
# seed.py re-stamps every record's timestamp so the newest lands ~5 min ago, identically
# for both backends — so time-windowed agent queries find data and the two arms stay
# byte-for-byte comparable. Source these, then call lvo_deploy_* + lvo_seed_*.
set -uo pipefail

LVO_NS=""
# Resolve this file's dir so dataset/seed paths work regardless of the case's CWD.
LVO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Deploy single-node Loki into a namespace and wait until the ingester is ready.
# Usage: lvo_deploy_loki <ns>
lvo_deploy_loki() {
  LVO_NS="$1"
  kubectl create namespace "$LVO_NS" --dry-run=client -o yaml | kubectl apply -f -
  kubectl label namespace "$LVO_NS" admission.datadoghq.com/enabled=false --overwrite
  kubectl apply -f "$LVO_DIR/../loki.yaml" -n "$LVO_NS"

  echo "⏳ Waiting for Loki pod to be ready (timeout 300s)..."
  local ready=false
  for i in $(seq 1 60); do
    if kubectl wait --for=condition=ready pod -l app=loki -n "$LVO_NS" --timeout=5s 2>/dev/null; then
      ready=true; break
    fi
    sleep 5
  done
  [ "$ready" = true ] || { echo "❌ Loki pod not ready"; kubectl get pods -n "$LVO_NS"; exit 1; }

  echo "⏳ Waiting for Loki ingester /ready..."
  for i in $(seq 1 60); do
    if kubectl exec -n "$LVO_NS" deployment/loki -- wget -q -O- http://localhost:3100/ready 2>/dev/null | grep -q "ready"; then
      echo "✅ Loki ready"; return 0
    fi
    sleep 2
  done
  echo "❌ Loki ingester not ready"; exit 1
}

# Deploy single-node OpenSearch into a namespace and wait for API health.
# Usage: lvo_deploy_opensearch <ns>
lvo_deploy_opensearch() {
  LVO_NS="$1"
  kubectl create namespace "$LVO_NS" --dry-run=client -o yaml | kubectl apply -f -
  kubectl apply -f "$LVO_DIR/../opensearch.yaml" -n "$LVO_NS"

  echo "⏳ Waiting for OpenSearch pod to be ready (timeout 300s)..."
  local ready=false
  for i in $(seq 1 60); do
    if kubectl wait --for=condition=ready pod -l app=opensearch -n "$LVO_NS" --timeout=5s 2>/dev/null; then
      ready=true; break
    fi
    sleep 5
  done
  [ "$ready" = true ] || { echo "❌ OpenSearch pod not ready"; kubectl get pods -n "$LVO_NS"; exit 1; }

  echo "⏳ Waiting for OpenSearch API health..."
  for i in $(seq 1 60); do
    local status
    status=$(kubectl exec -n "$LVO_NS" deployment/opensearch -- \
      curl -sf "http://localhost:9200/_cluster/health" 2>/dev/null | grep -oE '"status":"[a-z]+"' | cut -d'"' -f4)
    if [ "$status" = "green" ] || [ "$status" = "yellow" ]; then
      echo "✅ OpenSearch healthy (status=$status)"; return 0
    fi
    sleep 3
  done
  echo "❌ OpenSearch API not healthy"; exit 1
}

# Seed a dataset into Loki via cluster DNS. Usage: lvo_seed_loki <ns> <dataset>
lvo_seed_loki() {
  local ns="$1" dataset="$2"
  python3 "$LVO_DIR/seed.py" loki "$LVO_DIR/datasets/${dataset}.json" \
    "http://loki.${ns}.svc.cluster.local:3100" "$ns"
}

# Seed a dataset into OpenSearch via cluster DNS. Usage: lvo_seed_opensearch <ns> <dataset> <index>
lvo_seed_opensearch() {
  local ns="$1" dataset="$2" index="$3"
  python3 "$LVO_DIR/seed.py" opensearch "$LVO_DIR/datasets/${dataset}.json" \
    "http://opensearch.${ns}.svc.cluster.local:9200" "$index"
}
