#!/usr/bin/env python3
"""Seed one shared log dataset into Loki OR OpenSearch — identical content, so the
only variable in the comparison is the backend (+ its HolmesGPT toolset).

Usage:
  seed.py loki       <dataset.json> <loki_base_url> <namespace>
  seed.py opensearch <dataset.json> <os_base_url>   <index>

Dataset: JSON list of records {ts, level, service, pod, message, ...extra}. ts is
ISO8601. Timestamps are RE-STAMPED at load so the newest record is ~5 min ago
(relative spacing preserved), so time-windowed agent queries find the data regardless
of when the benchmark runs — same fix used for the trace cases.

Deterministic: no randomness; re-stamp shift is the only time-dependent part, applied
identically to both backends in a run.
"""
import json
import sys
import urllib.request
import datetime


def restamp(records):
    times = [datetime.datetime.fromisoformat(r["ts"].replace("Z", "+00:00")) for r in records]
    shift = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=5) - max(times)
    for r, t in zip(records, times):
        r["_t"] = t + shift
    return records


def post(url, data, headers, method="POST"):
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.status, resp.read().decode()


def seed_loki(records, base, namespace):
    # Group into streams by (level, service, pod); push JSON log lines, ts in nanos.
    streams = {}
    for r in records:
        key = (r["level"], r.get("service", ""), r.get("pod", ""))
        streams.setdefault(key, [])
        line = json.dumps({k: v for k, v in r.items() if k != "_t"})
        ns = str(int(r["_t"].timestamp() * 1e9))
        streams[key].append([ns, line])
    payload = {"streams": [
        {"stream": {"namespace": namespace, "level": lv, "service": sv, "pod": pod},
         "values": vals}
        for (lv, sv, pod), vals in streams.items()
    ]}
    st, body = post(f"{base}/loki/api/v1/push", json.dumps(payload).encode(),
                    {"Content-Type": "application/json"})
    print(f"loki push: {st} ({len(records)} lines, {len(streams)} streams)")


def seed_opensearch(records, base, index):
    mapping = {"settings": {"number_of_shards": 1, "number_of_replicas": 0},
               "mappings": {"properties": {
                   "timestamp": {"type": "date"}, "level": {"type": "keyword"},
                   "service": {"type": "keyword"}, "pod": {"type": "keyword"},
                   "error_code": {"type": "keyword"}, "message": {"type": "text"}}}}
    try:
        post(f"{base}/{index}", json.dumps(mapping).encode(),
             {"Content-Type": "application/json"}, method="PUT")
    except Exception as e:
        print(f"(index create: {e})")
    lines = []
    for r in records:
        doc = {k: v for k, v in r.items() if k not in ("_t", "ts")}
        doc["timestamp"] = r["_t"].strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        lines.append('{"index":{}}')
        lines.append(json.dumps(doc))
    body = ("\n".join(lines) + "\n").encode()
    st, _ = post(f"{base}/{index}/_bulk", body, {"Content-Type": "application/x-ndjson"})
    post(f"{base}/{index}/_refresh", b"", {})
    print(f"opensearch bulk: {st} ({len(records)} docs -> {index})")


def main():
    backend, path = sys.argv[1], sys.argv[2]
    records = restamp(json.load(open(path)))
    if backend == "loki":
        seed_loki(records, sys.argv[3], sys.argv[4])
    elif backend == "opensearch":
        seed_opensearch(records, sys.argv[3], sys.argv[4])
    else:
        print("backend must be loki|opensearch"); sys.exit(1)


if __name__ == "__main__":
    main()
