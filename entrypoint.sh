#!/bin/sh

echo "Waiting for Cassandra ring..."

# Use cqlsh to check local system status instead of nodetool which requires full classpaths
until cqlsh cassandra 9042 -e "SELECT release_version FROM system.local" > /dev/null 2>&1
do
  echo "Node not Up/Normal yet..."
  sleep 5
done

# Secondary wait for gossip/token agreement
until cqlsh cassandra 9042 -e "SELECT schema_version FROM system.local WHERE key='local'" > /dev/null 2>&1
do
  echo "Schema not agreed yet..."
  sleep 5
done

echo "Cassandra ring ready."

echo "Applying schema..."
cqlsh cassandra 9042 -f /schema/schema.cql

echo "Waiting for epistemic_runtime keyspace..."
until cqlsh cassandra 9042 -e "USE epistemic_runtime" > /dev/null 2>&1
do
  echo "Keyspace not ready..."
  sleep 3
done

echo "Cassandra fully ready."
echo "Starting service..."

exec "$@"
