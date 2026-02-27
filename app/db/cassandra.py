import threading
import time
import os

from cassandra.cluster import Cluster

_lock = threading.Lock()
_session = None
_cluster = None

CASSANDRA_HOST = os.getenv("CASSANDRA_HOST", "cassandra")
CASSANDRA_KEYSPACE = os.getenv("CASSANDRA_KEYSPACE", "epistemic_runtime")


def get_session():
    """
    Thread-safe lazy singleton session.

    Double-checked locking: the outer check avoids acquiring the lock on
    every call once the session exists; the inner check prevents a race
    between two threads that both saw _session is None simultaneously.
    """
    global _session, _cluster

    if _session is not None:
        return _session

    with _lock:
        if _session is not None:          # Re-check after acquiring lock
            return _session

        for attempt in range(1, 31):
            try:
                print(f"Cassandra connect attempt {attempt}/30 ...")
                _cluster = Cluster([CASSANDRA_HOST])
                _session = _cluster.connect(CASSANDRA_KEYSPACE)
                print("Cassandra connected.")
                return _session
            except Exception as exc:
                print(f"Cassandra not ready: {exc}. Retrying in 3 s ...")
                time.sleep(3)

        raise RuntimeError("Cassandra never became ready after 30 attempts")


def close_session():
    """Graceful shutdown — call from application teardown."""
    global _session, _cluster
    with _lock:
        if _session is not None:
            _session.shutdown()
            _session = None
        if _cluster is not None:
            _cluster.shutdown()
            _cluster = None