"""Exercise the production asynchronous sink through a shared OS pipe."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "python/sglang/srt/observability/pd_lifecycle.py"
spec = importlib.util.spec_from_file_location("pd_wire", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

if __name__ != "__main__":
    from sglang.test.ci.ci_register import register_cpu_ci

    register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PRODUCER = """
import importlib.util, json, sys
from types import SimpleNamespace
spec=importlib.util.spec_from_file_location('actual_pd',sys.argv[1])
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
r=m.PDLifecycleRecorder(10,max_events=8192)
for batch in range(4):
    for i in range(300):
        q=SimpleNamespace(rid=f'{batch}-{i}',bootstrap_room=2**64-1,time_stats=SimpleNamespace())
        r.transition(q,'arrival')
        r.transition(q,'handshake_ready')
        r.transition(q,'destination_published')
        r.transition(q,'transfer_complete')
        r.transition(q,'completion')
    r.summary({'transfer':0},force=True)
    r._sink.join()
r.close()
assert not r.loss, dict(r.loss)
"""


class TestDelivery(unittest.TestCase):
    def frames(self, number=1, **extra):
        record = dict(context=dict(incarnation="a" * 32), events=[], **extra)
        with tempfile.TemporaryFile() as f:
            module.write_atomic_record(record, number, f.fileno())
            f.seek(0)
            return f.read().decode().splitlines(keepends=True)

    def test_concurrent_actual_sink_and_ordinary_output(self):
        read_fd, write_fd = os.pipe()
        chunks = []

        def drain():
            with os.fdopen(read_fd, "rb") as stream:
                while chunk := stream.read(65536):
                    chunks.append(chunk)

        reader = threading.Thread(target=drain)
        reader.start()
        processes = [
            subprocess.Popen(
                [sys.executable, "-c", PRODUCER, str(SOURCE)],
                stderr=write_fd,
                stdout=subprocess.DEVNULL,
            )
            for _ in range(4)
        ]
        noise = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import os; [(os.write(2,b'INFO '+b'x'*8192),os.write(2,b' HTTP 200\\n')) for _ in range(300)]",
            ],
            stderr=write_fd,
        )
        os.close(write_fd)
        for process in processes + [noise]:
            self.assertEqual(process.wait(timeout=30), 0)
        reader.join(timeout=30)
        self.assertFalse(reader.is_alive())
        raw = b"".join(chunks)
        decoder = module.PDFrameReader()
        rows = []
        for i, line in enumerate(raw.decode().splitlines(keepends=True)):
            result = decoder.feed(line, dict(line=i))
            if result:
                rows.append(result[0])
        self.assertEqual(decoder.finish()["gaps"], [])
        self.assertEqual(len(rows), 20)  # Four batches and one shutdown per process.
        self.assertEqual(len({r["context"]["incarnation"] for r in rows}), 4)
        self.assertEqual(sum(len(r["events"]) for r in rows), 4 * 4 * 300 * 5)
        self.assertTrue(all(not r["loss_total"] for r in rows))
        self.assertTrue(
            all(e["room"] == str(2**64 - 1) for r in rows for e in r["events"])
        )
        print(
            json.dumps(
                dict(
                    shared_pipe_bytes=len(raw),
                    records=len(rows),
                    events=sum(len(r["events"]) for r in rows),
                    transport_gaps=decoder.gaps,
                )
            )
        )

    def test_fragment_loss_corruption_and_capture_boundary(self):
        lines = self.frames(padding="x" * 10000)
        decoder = module.PDFrameReader()
        for line in lines[:1] + lines[2:]:
            decoder.feed(line)
        self.assertTrue(decoder.finish()["gaps"])
        decoder = module.PDFrameReader()
        decoder.feed(lines[0])
        self.assertEqual(
            decoder.finish()["gaps"][0]["reason"],
            "incomplete_record_at_capture_boundary",
        )
        decoder = module.PDFrameReader()
        bad = json.loads(lines[0].split(module.WIRE_PREFIX)[1])
        bad["p"] = "!"
        self.assertIsNone(decoder.feed(module.WIRE_PREFIX + json.dumps(bad)))
        self.assertEqual(decoder.gaps[0]["reason"], "invalid_atomic_frame")
        decoder = module.PDFrameReader()
        for line in self.frames(2):
            decoder.feed(line)
        self.assertEqual(decoder.gaps[0]["reason"], "record_sequence_gap_or_reset")

    def test_reader_bounds_and_sink_failure_accounting(self):
        decoder = module.PDFrameReader(max_pending_bytes=10)
        for line in self.frames():
            self.assertIsNone(decoder.feed(line))
        self.assertEqual(decoder.pending_bytes, 0)
        self.assertEqual(decoder.gaps[0]["reason"], "reader_byte_bound_exceeded")
        decoder = module.PDFrameReader(max_processes=0)
        for line in self.frames():
            self.assertIsNone(decoder.feed(line))
        self.assertFalse(decoder.pending)
        for _ in range(200):
            decoder.feed(module.WIRE_PREFIX + "x" * 5000)
        self.assertEqual(len(decoder.gaps), 128)
        self.assertGreater(decoder.gaps_omitted, 0)
        with patch.object(
            module, "write_atomic_record", side_effect=OSError("broken sink")
        ):
            rec = module.PDLifecycleRecorder(10)
            rec.summary({}, force=True)
            rec._sink.join()
            rec.close()
            self.assertEqual(rec.loss["sink_errors"], 2)
        with patch.object(module.os, "write", return_value=0):
            with tempfile.TemporaryFile() as f, self.assertRaises(OSError):
                module.write_atomic_record(
                    dict(context=dict(incarnation="a" * 32)), 1, f.fileno()
                )


if __name__ == "__main__":
    if "--emit-native-shared-output" in sys.argv:
        # Inherit the actual SRT/Slurm capture descriptor. The external watcher
        # verifies every record after that step closes, not just an inner pipe.
        processes = [
            subprocess.Popen([sys.executable, "-c", PRODUCER, str(SOURCE)])
            for _ in range(4)
        ]
        noise = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import os; [(os.write(2,b'INFO '+b'x'*8192),os.write(2,b' HTTP 200\\n')) for _ in range(300)]",
            ]
        )
        for process in processes + [noise]:
            if process.wait(timeout=30) != 0:
                raise SystemExit("Native shared-output producer failed")
    else:
        unittest.main()
