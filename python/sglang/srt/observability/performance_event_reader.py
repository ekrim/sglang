"""Bounded consumer for mixed stderr performance events; absence requires closure.

python performance_event_reader.py LOG [LOG ...]
Outputs episode summaries and coverage, never promotes a causal explanation.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import Counter

PREFIX = "performance_event "
EVENT_NAMES = {"queue.initial", "queue.pressure_enter", "queue.pressure_exit",
               "arrival.window", "arrival.surge_enter", "arrival.surge_exit",
               "cache.device_evict", "cache.host_evict", "cache.backup_complete",
               "cache.loadback_complete", "decode.retraction", "decision.changed"}


class EventReader:
    def __init__(self, max_processes=128, max_episodes=1000):
        self.max_processes, self.max_episodes = max_processes, max_episodes
        self.streams = {}
        self.gaps = Counter()
        self.counts = Counter()
        self.episodes = []
        self.episodes_omitted = 0

    def feed(self, line):
        if PREFIX not in line:
            return None
        try:
            record = json.loads(line.split(PREFIX, 1)[1])
            context = record["context"]
            incarnation = context["incarnation"]
            number = record["delivery_seq"]
            assert record["schema_version"] == 1 and type(number) is int and number > 0
            assert type(incarnation) is str and len(incarnation) <= 128
            assert record["kind"] in {"event", "heartbeat", "closed"}
            if record["kind"] == "event":
                assert type(record["seq"]) is int and record["seq"] > 0
                assert type(record["event"]) is str and record["event"] in EVENT_NAMES
                assert type(record["occurred_monotonic"]) in (int, float) and math.isfinite(record["occurred_monotonic"])
                assert type(record["values"]) is dict and len(record["values"]) <= 8
            else:
                assert type(record["loss_total"]) is dict and len(record["loss_total"]) <= 16
                assert all(type(v) is int and v >= 0 for v in record["loss_total"].values())
                assert type(record["last_event_seq"]) is int and record["last_event_seq"] >= 0
        except (ValueError, KeyError, TypeError, AssertionError):
            self.gaps["malformed"] += 1
            return None
        if incarnation not in self.streams:
            if len(self.streams) >= self.max_processes:
                self.gaps["process_limit"] += 1
                return None
            self.streams[incarnation] = dict(context=context, delivery_seq=0, event_seq=0,
                                            closed=False, loss_total={}, gaps=0, active={})
        stream = self.streams[incarnation]
        if number != stream["delivery_seq"] + 1:
            stream["gaps"] += 1
        if stream["closed"]:
            stream["gaps"] += 1
        stream["delivery_seq"] = number
        if record["kind"] != "event":
            stream["loss_total"] = record.get("loss_total", {})
            stream["last_event_seq"] = record.get("last_event_seq")
            stream["closed"] = record["kind"] == "closed"
            if stream["closed"] and stream["last_event_seq"] != stream["event_seq"]:
                stream["gaps"] += 1
            return record
        if record["seq"] != stream["event_seq"] + 1:
            stream["gaps"] += 1
        stream["event_seq"] = record["seq"]
        name, values = record["event"], record["values"]
        self.counts[name] += 1
        if name in {"queue.pressure_enter", "arrival.surge_enter"}:
            key = values.get("queue", "arrival")
            if key not in {"execution_wait", "bootstrap", "preallocation", "transfer", "retracted", "inflight", "arrival"}:
                self.gaps["unknown_episode"] += 1
                return record
            if key in stream["active"]:
                stream["gaps"] += 1
            stream["active"][key] = dict(episode=values.get("episode"), start=record["occurred_monotonic"])
        elif name in {"queue.pressure_exit", "arrival.surge_exit"}:
            key = values.get("queue", "arrival")
            start = stream["active"].pop(key, None)
            episode = dict(incarnation=incarnation, condition=key, values=values,
                           start=start, end=record["occurred_monotonic"],
                           paired=start is not None and start["episode"] == values.get("episode"))
            if not episode["paired"]:
                stream["gaps"] += 1
            if len(self.episodes) < self.max_episodes:
                self.episodes.append(episode)
            else:
                self.episodes_omitted += 1
        return record

    def report(self):
        streams = []
        for incarnation, stream in self.streams.items():
            streams.append(dict(incarnation=incarnation, **stream,
                                coverage=("closed_contiguous" if stream["closed"] and
                                          not stream["gaps"] and not any(stream["loss_total"].values())
                                          and not self.gaps else "incomplete_or_live")))
        return dict(schema_version=1, streams=streams, transport_gaps=dict(self.gaps),
                    event_counts=dict(self.counts), episodes=self.episodes,
                    episodes_omitted=self.episodes_omitted,
                    interpretation="Temporal candidates only. Missing producer, live tail, loss, or unpaired episode is unknown, not zero. Queue recovery does not imply requests completed. Join same-incarnation PD holders/stages; compare matched workload windows and counterexamples before revising explanations.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+")
    args = parser.parse_args()
    reader = EventReader()
    for path in args.logs:
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt", errors="replace") as handle:
            while True:
                line = handle.readline(16384)
                if not line:
                    break
                if len(line) == 16384 and not line.endswith("\n"):
                    reader.gaps["oversized_input_line"] += 1
                    while line and not line.endswith("\n"):
                        line = handle.readline(16384)
                    continue
                reader.feed(line)
    print(json.dumps(reader.report(), indent=2))


if __name__ == "__main__":
    main()
