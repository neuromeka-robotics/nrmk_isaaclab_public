import gc
import multiprocessing as mp
import os
import pickle
import statistics as stats
import struct
import time
from typing import Any, Callable, Dict, Tuple  # noqa: F401

import msgpack
import numpy as np

# -----------------------------
# Configuration
# -----------------------------
# Image: 640x480 float32 (~1.2MB raw, depends on packing)
ARR = np.zeros((1, 480, 640, 3), np.float32)
RAW_IMAGE = ARR.tobytes()
META = {"dtype": "float32", "shape": [1, 480, 640, 3]}


# -----------------------------
# Serialization Helpers
# -----------------------------
def pack_raw(ts_ns: int) -> bytes:
    return struct.pack("<Q", ts_ns) + RAW_IMAGE


def unpack_raw(b: bytes) -> Tuple[int, bytes]:
    ts_ns = struct.unpack("<Q", b[:8])[0]
    return ts_ns, b[8:]


def pack_msgpack(ts_ns: int) -> bytes:
    return msgpack.packb({"m": META, "t": ts_ns, "d": RAW_IMAGE}, use_bin_type=True)


def unpack_msgpack(b: bytes) -> Tuple[int, bytes]:
    obj = msgpack.unpackb(b, raw=False)
    return int(obj["t"]), obj["d"]


def pack_pickle(ts_ns: int) -> bytes:
    return pickle.dumps({"m": META, "t": ts_ns, "d": RAW_IMAGE}, protocol=pickle.HIGHEST_PROTOCOL)


def unpack_pickle(b: bytes) -> Tuple[int, bytes]:
    obj = pickle.loads(b)
    return int(obj["t"]), obj["d"]


CODECS = {
    "raw": (pack_raw, unpack_raw),
    "msgpack": (pack_msgpack, unpack_msgpack),
    "pickle": (pack_pickle, unpack_pickle),
}

# -----------------------------
# Worker Processes
# -----------------------------


def set_cpu_affinity(core_id: int):
    """Pin this process to a specific core (Linux/Ubuntu only)."""
    try:
        os.sched_setaffinity(0, {core_id})
    except AttributeError:
        pass  # Not on Linux


def server_process(transport: str, pattern: str, codec: str, stop_event: mp.Event):
    """
    Independent Server Process.
    """
    # Pin to Core 2 (arbitrary choice to separate from client)
    set_cpu_affinity(2)

    _, unpack_fn = CODECS[codec]

    if transport == "zmq":
        import zmq

        ctx = zmq.Context()

        if pattern == "reqrep":
            socket = ctx.socket(zmq.REP)
            socket.bind("tcp://127.0.0.1:5555")
            while not stop_event.is_set():
                if socket.poll(100):  # Check every 100ms
                    msg = socket.recv()
                    _ = unpack_fn(msg)
                    socket.send(msg)  # Echo

        elif pattern == "pubsub":
            # Echo server: SUB to 'req', PUB to 'rep'
            sub = ctx.socket(zmq.SUB)
            sub.bind("tcp://127.0.0.1:5555")
            sub.setsockopt(zmq.SUBSCRIBE, b"")

            pub = ctx.socket(zmq.PUB)
            pub.bind("tcp://127.0.0.1:5556")

            while not stop_event.is_set():
                if sub.poll(100):
                    msg = sub.recv()
                    _ = unpack_fn(msg)
                    pub.send(msg)

        ctx.destroy()

    elif transport == "zenoh":
        import zenoh

        # Force TCP to ensure we test network stack, not internal SHM shortcuts
        conf = zenoh.Config()

        # NOTE: Opening Zenoh inside the process is crucial
        session = zenoh.open(conf)
        key_req = "bench/req"
        key_rep = "bench/rep"

        if pattern == "pubsub":

            def listener(sample):
                b = sample.payload.to_bytes()
                _ = unpack_fn(b)
                session.put(key_rep, b)

            sub = session.declare_subscriber(key_req, listener)
            stop_event.wait()
            sub.undeclare()

        elif pattern == "query":
            # Queryable: receives GET, replies with payload
            def on_query(query):
                # Retrieve payload sent with query if exists, else generic
                b = query.payload.to_bytes()
                _ = unpack_fn(b)
                query.reply(query.key_expr, b)

            qbl = session.declare_queryable(key_req, on_query)
            stop_event.wait()
            qbl.undeclare()

        session.close()


def client_process(transport: str, pattern: str, codec: str, iters: int, warmup: int, result_queue: mp.Queue):
    """
    Independent Client Process.
    """
    # Pin to Core 4
    set_cpu_affinity(4)

    pack_fn, unpack_fn = CODECS[codec]
    rtts = []

    # 1. Setup Transport
    if transport == "zmq":
        import zmq

        ctx = zmq.Context()
        if pattern == "reqrep":
            socket = ctx.socket(zmq.REQ)
            socket.connect("tcp://127.0.0.1:5555")

            def do_rtt():
                payload = pack_fn(time.perf_counter_ns())
                t0 = time.perf_counter()
                socket.send(payload)
                resp = socket.recv()
                t1 = time.perf_counter()
                _ = unpack_fn(resp)
                return (t1 - t0) * 1000.0

        elif pattern == "pubsub":
            pub = ctx.socket(zmq.PUB)
            pub.connect("tcp://127.0.0.1:5555")

            sub = ctx.socket(zmq.SUB)
            sub.connect("tcp://127.0.0.1:5556")
            sub.setsockopt(zmq.SUBSCRIBE, b"")
            time.sleep(0.2)  # Wait for sub connection

            def do_rtt():
                payload = pack_fn(time.perf_counter_ns())
                t0 = time.perf_counter()
                pub.send(payload)
                resp = sub.recv()  # Blocking
                t1 = time.perf_counter()
                _ = unpack_fn(resp)
                return (t1 - t0) * 1000.0

    elif transport == "zenoh":
        import zenoh

        session = zenoh.open(zenoh.Config())
        key_req = "bench/req"
        key_rep = "bench/rep"

        if pattern == "pubsub":
            # For Zenoh PubSub RTT, we need a blocking approach for the client side
            # We'll use a promise/future pattern roughly
            import queue

            q = queue.Queue()

            def listener(sample):
                q.put(sample.payload.to_bytes())

            sub = session.declare_subscriber(key_rep, listener)

            def do_rtt():
                payload = pack_fn(time.perf_counter_ns())
                # Empty queue from potential old stuff
                while not q.empty():
                    q.get()

                t0 = time.perf_counter()
                session.put(key_req, payload)
                resp = q.get(timeout=2.0)
                t1 = time.perf_counter()
                _ = unpack_fn(resp)
                return (t1 - t0) * 1000.0

        elif pattern == "query":

            def do_rtt():
                payload = pack_fn(time.perf_counter_ns())
                t0 = time.perf_counter()
                # Send payload attached to the GET
                replies = session.get(key_req, payload=zenoh.ZBytes(payload))
                got_data = False
                for r in replies:
                    if r.ok:
                        _ = unpack_fn(r.ok.payload.to_bytes())
                        got_data = True
                        break
                t1 = time.perf_counter()
                return (t1 - t0) * 1000.0 if got_data else 0.0

    # 2. Warmup
    for _ in range(warmup):
        try:
            do_rtt()
        except Exception:
            pass

    # 3. Measurement Loop (GC Disabled)
    gc.collect()
    gc.disable()

    # start_time = time.perf_counter()
    try:
        for _ in range(iters):
            rtts.append(do_rtt())
    except Exception as e:
        print(f"Client Error: {e}")
    finally:
        gc.enable()

    # Teardown
    if transport == "zmq":
        ctx.destroy()
    elif transport == "zenoh":
        session.close()

    # 4. Report
    result_queue.put(rtts)


# -----------------------------
# Main Runner
# -----------------------------
def run_benchmark(transport, pattern, codec, iters=1000, warmup=20):
    stop_event = mp.Event()
    result_queue = mp.Queue()

    # Start Server
    p_server = mp.Process(target=server_process, args=(transport, pattern, codec, stop_event))
    p_server.start()

    # Allow server to bind/settle
    time.sleep(1.0)

    # Start Client
    p_client = mp.Process(target=client_process, args=(transport, pattern, codec, iters, warmup, result_queue))
    p_client.start()

    # Wait for client to finish
    p_client.join()

    # Stop Server
    stop_event.set()
    p_server.join()

    if not result_queue.empty():
        rtts = result_queue.get()
        if not rtts:
            return None

        mean = stats.mean(rtts)
        p95 = sorted(rtts)[int(len(rtts) * 0.95)]

        # Calculate approximate throughput (MiB/s)
        # Payload size approx
        if codec == "raw":
            p_size = len(RAW_IMAGE) + 8
        elif codec == "msgpack":
            p_size = len(RAW_IMAGE) + 100  # Overhead estimate
        else:
            p_size = len(RAW_IMAGE) + 150

        # throughput = (bytes) / (RTT/2 in seconds)
        one_way_sec = (mean / 1000.0) / 2.0
        tput = (p_size / (1024 * 1024)) / one_way_sec

        print(
            f"{transport:6s} | {pattern:6s} | {codec:7s} | RTT: {mean:.2f}ms (p95: {p95:.2f}) | T-put: {tput:.1f} MiB/s"
        )
    else:
        print(f"Failed to get results for {transport} {pattern}")


def main():
    # Try to increase priority, but ignore if not allowed (no sudo)
    # try:
    #     os.nice(-10)
    #     print("High process priority set.")
    # except PermissionError:
    #     print("Running with standard priority (sudo not available).")
    # except AttributeError:
    #     pass

    print(f"Benchmarking with Multiprocessing (PID: {os.getpid()})")
    print("Transport | Pattern | Codec   | RTT Mean  | Throughput (Est)")
    print("-" * 65)

    configs = [
        ("zmq", "reqrep"),
        ("zmq", "pubsub"),
        ("zenoh", "pubsub"),
        ("zenoh", "query"),
    ]

    for trans, pat in configs:
        for codec in ["raw", "msgpack", "pickle"]:
            run_benchmark(trans, pat, codec, iters=5000, warmup=50)
            time.sleep(0.5)  # Cooldown


if __name__ == "__main__":
    main()
