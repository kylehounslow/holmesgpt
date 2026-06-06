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
    # Create a Dashboards index pattern <index>* so TestPulsar discovers it. Anonymous
    # read can't write saved objects, so use admin basic auth against the OSD API.
    kubectl exec -n "$TP_NS" deploy/"$TP_OSD" -- sh -c '
      P="${OPENSEARCH_INITIAL_ADMIN_PASSWORD:-My_password_123!@#}"
      curl -s -u admin:"$P" -X POST "http://localhost:5601/api/saved_objects/index-pattern/'"$index"'-pattern" \
        -H "osd-xsrf: true" -H "Content-Type: application/json" \
        -d "{\"attributes\":{\"title\":\"'"$index"'*\",\"timeFieldName\":\"timestamp\"}}" \
        >/dev/null 2>&1 && echo "✅ created index pattern '"$index"'*" || echo "(index-pattern create returned non-200; may already exist)"
    '
  fi
}
