#!/bin/bash
# Schema-aware tp-stack seeder for the schema-bias experiment (does TestPulsar's tooling
# bias the agent toward the OTel-logs schema regardless of the actual data?).
#
# Seeds the SAME committed dataset into the shared tp-stack OpenSearch in one of two
# field schemas, and (for a neutrally-named index) creates the Dashboards index pattern
# so TestPulsar's list_index_patterns surfaces it.
#
#   lvo_seed_schema <dataset> <index> <schema:otel|flat> [make_pattern:0|1]
#
# schema=otel : real OTel-logs fields (body, severityText, @timestamp, serviceName) —
#               what data-prepper writes and what TestPulsar's SOP lore describes.
# schema=flat : our synthetic fields (message, level, timestamp, service).
# make_pattern=1 : POST a Dashboards index-pattern for <index>* so it is discoverable
#               even when the name doesn't match a pre-existing otel pattern.
set -uo pipefail

LVO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TP_NS="${TP_NS:-tp-stack}"
TP_POD="${TP_POD:-opensearch-cluster-master-0}"
TP_OSD="${TP_OSD:-obs-opensearch-dashboards}"

lvo_seed_schema() {
  local dataset="$1" index="$2" schema="$3" make_pattern="${4:-0}"
  local bulk=/tmp/tp_${dataset}_${schema}_$$.ndjson

  python3 - "$LVO_DIR/datasets/${dataset}.json" "$schema" > "$bulk" <<'PY'
import json, sys, datetime
records = json.load(open(sys.argv[1])); schema = sys.argv[2]
times = [datetime.datetime.fromisoformat(r["ts"].replace("Z","+00:00")) for r in records]
shift = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=5) - max(times)
for r, t in zip(records, times):
    ts = (t + shift).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    if schema == "otel":
        # Real OTel-logs shape (data-prepper logs-otel-v1): body / severityText /
        # @timestamp / serviceName. This is what TestPulsar's defaults assume.
        doc = {"@timestamp": ts, "time": ts, "body": r.get("message",""),
               "severityText": r.get("level",""), "serviceName": r.get("service",""),
               "resource.attributes.k8s.pod.name": r.get("pod","")}
    else:
        # Our synthetic flat shape: message / level / timestamp / service.
        doc = {k: v for k, v in r.items() if k != "ts"}
        doc["timestamp"] = ts
    print('{"index":{}}'); print(json.dumps(doc, separators=(",", ":")))
PY

  kubectl exec -i -n "$TP_NS" "$TP_POD" -c opensearch -- sh -c '
    P="$OPENSEARCH_INITIAL_ADMIN_PASSWORD"
    curl -sk -u admin:"$P" -X DELETE "https://localhost:9200/'"$index"'" >/dev/null 2>&1 || true
    curl -sk -u admin:"$P" -X POST "https://localhost:9200/'"$index"'/_bulk?refresh=true" \
      -H "Content-Type: application/x-ndjson" --data-binary @- | grep -q "\"errors\":false" \
      && echo "✅ tp-stack bulk ok ('"$schema"')" || { echo "❌ tp-stack bulk had errors"; exit 1; }
  ' < "$bulk"
  rm -f "$bulk"

  if [ "$make_pattern" = "1" ]; then
    # Create a Dashboards index pattern <index>* so TestPulsar discovers it. It MUST live
    # in the same workspace TestPulsar scopes its list_index_patterns/_find to, or the
    # agent never sees it (a pattern created globally with no workspace is invisible to
    # the workspace-scoped query). Discover the workspace id by name ("Observability
    # Stack", created by the chart) and create the pattern via the /w/<id>/ API path.
    # Anonymous read can't write saved objects, so use admin basic auth.
    kubectl exec -n "$TP_NS" deploy/"$TP_OSD" -- sh -c '
      P="${OPENSEARCH_INITIAL_ADMIN_PASSWORD:-My_password_123!@#}"
      WS=$(curl -s -u admin:"$P" -X POST "http://localhost:5601/api/workspaces/_list" \
        -H "osd-xsrf: true" -H "Content-Type: application/json" -d "{}" \
        | python3 -c "import sys,json; ws=json.load(sys.stdin).get(\"result\",{}).get(\"workspaces\",[]); print(next((w[\"id\"] for w in ws if \"bservability\" in w.get(\"name\",\"\")), ws[0][\"id\"] if ws else \"\"))")
      [ -n "$WS" ] || { echo "❌ no workspace found for index-pattern"; exit 1; }
      # Scrub any stale copy first (a prior run may have died before after_test), so the
      # create is clean and we never accumulate duplicate/empty patterns that mislead the
      # agent. Both workspace-scoped and global, ignore failures.
      curl -s -u admin:"$P" -X DELETE "http://localhost:5601/w/$WS/api/saved_objects/index-pattern/'"$index"'-pattern" -H "osd-xsrf: true" >/dev/null 2>&1
      curl -s -u admin:"$P" -X DELETE "http://localhost:5601/api/saved_objects/index-pattern/'"$index"'-pattern" -H "osd-xsrf: true" >/dev/null 2>&1
      curl -s -u admin:"$P" -X POST "http://localhost:5601/w/$WS/api/saved_objects/index-pattern/'"$index"'-pattern" \
        -H "osd-xsrf: true" -H "Content-Type: application/json" \
        -d "{\"attributes\":{\"title\":\"'"$index"'*\",\"timeFieldName\":\"timestamp\"}}" \
        >/dev/null 2>&1 && echo "✅ created index pattern '"$index"'* in workspace $WS" \
        || echo "(index-pattern create returned non-200; may already exist)"
    '
  fi
}

# Tear down a bench index AND its Dashboards index pattern from the shared tp-stack, so
# a leftover empty pattern can't mislead a later MCP run's list_index_patterns (the same
# stale-pattern artifact that caused the original 38% MCP result). Idempotent: safe to
# call even if the index/pattern is already gone, or this run never created a pattern.
# Call from after_test. Usage: lvo_teardown_tpstack <index>
lvo_teardown_tpstack() {
  local index="$1"
  kubectl exec -n "$TP_NS" "$TP_POD" -c opensearch -- sh -c \
    'curl -sk -u admin:"$OPENSEARCH_INITIAL_ADMIN_PASSWORD" -X DELETE "https://localhost:9200/'"$index"'"' \
    >/dev/null 2>&1 || true
  kubectl exec -n "$TP_NS" deploy/"$TP_OSD" -- sh -c '
    P="${OPENSEARCH_INITIAL_ADMIN_PASSWORD:-My_password_123!@#}"
    WS=$(curl -s -u admin:"$P" -X POST "http://localhost:5601/api/workspaces/_list" \
      -H "osd-xsrf: true" -H "Content-Type: application/json" -d "{}" \
      | python3 -c "import sys,json; ws=json.load(sys.stdin).get(\"result\",{}).get(\"workspaces\",[]); print(next((w[\"id\"] for w in ws if \"bservability\" in w.get(\"name\",\"\")), ws[0][\"id\"] if ws else \"\"))" 2>/dev/null)
    # Delete the pattern both workspace-scoped (where we create it) and globally (in case
    # an earlier buggy run left a ws=None copy). Ignore failures.
    [ -n "$WS" ] && curl -s -u admin:"$P" -X DELETE "http://localhost:5601/w/$WS/api/saved_objects/index-pattern/'"$index"'-pattern" -H "osd-xsrf: true" >/dev/null 2>&1
    curl -s -u admin:"$P" -X DELETE "http://localhost:5601/api/saved_objects/index-pattern/'"$index"'-pattern" -H "osd-xsrf: true" >/dev/null 2>&1
    true
  ' >/dev/null 2>&1 || true
  echo "🧹 tore down tp-stack index + pattern: '"$index"'"
}
