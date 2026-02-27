import os
os.environ["CASSANDRA_HOST"] = "localhost"
from app.db.cassandra import get_session

def check():
    session = get_session()
    # Let's get the primary key specifically
    rows = session.execute("SELECT column_name, kind, position FROM system_schema.columns WHERE keyspace_name = 'epistemic_runtime' AND table_name = 'opinions_by_claim' ALLOW FILTERING")
    
    partition_keys = []
    clustering_keys = []
    
    for row in rows:
        if row.kind == 'partition_key':
            partition_keys.append((row.position, row.column_name))
        elif row.kind == 'clustering':
            clustering_keys.append((row.position, row.column_name))
            
    partition_keys.sort()
    clustering_keys.sort()
    
    print("Partition Keys:", [k[1] for k in partition_keys])
    print("Clustering Keys:", [k[1] for k in clustering_keys])

check()
