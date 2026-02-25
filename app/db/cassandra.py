from cassandra.cluster import Cluster
import time
import os

_session = None
CASSANDRA_HOST = os.getenv("CASSANDRA_HOST", "cassandra")

def get_session():
    global _session

    if _session is not None:
        return _session

    for i in range(30):
        try:
            print(f"Trying Cassandra connect attempt {i+1}/30...")
            cluster = Cluster([CASSANDRA_HOST])
            _session = cluster.connect("epistemic_runtime")
            print("Cassandra Connected!")
            return _session
        except Exception as e:
            print(f"Cassandra not ready: {e}. Retrying...")
            time.sleep(3)

    raise Exception("Cassandra never became ready")
