"""
conftest.py
-----------
Pytest configuration: stubs out all external dependencies (Cassandra,
Redis, Celery, structlog) before any test module is imported.

This runs automatically — no imports needed in test files.
"""

import sys
import types
from collections import namedtuple


def _make_module(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


# ---------------------------------------------------------------------------
# cassandra
# ---------------------------------------------------------------------------
_make_module("cassandra")
_make_module("cassandra.cluster", Cluster=object)
_make_module("cassandra.concurrent", execute_concurrent=lambda *a, **k: [])
_CL = type("ConsistencyLevel", (), {"QUORUM": 2, "SERIAL": 3})()
_make_module("cassandra.query", BatchStatement=object, ConsistencyLevel=_CL)

# ---------------------------------------------------------------------------
# structlog
# ---------------------------------------------------------------------------
_NoopLogger = type("_NoopLogger", (), {
    "info":    lambda self, *a, **k: None,
    "error":   lambda self, *a, **k: None,
    "warning": lambda self, *a, **k: None,
    "debug":   lambda self, *a, **k: None,
})
_make_module("structlog", get_logger=lambda: _NoopLogger())

# ---------------------------------------------------------------------------
# redis
# ---------------------------------------------------------------------------
_make_module("redis.exceptions", LockNotOwnedError=Exception)
_make_module("redis", Redis=lambda **k: None,
             exceptions=sys.modules["redis.exceptions"])

# ---------------------------------------------------------------------------
# celery
# ---------------------------------------------------------------------------
def _noop_decorator(*args, **kwargs):
    """Works as @celery.task and @celery.task(bind=True, ...)"""
    if len(args) == 1 and callable(args[0]):
        return args[0]
    return lambda f: f

_FakeCelery = type("FakeCelery", (), {"task": staticmethod(_noop_decorator)})
_make_module("celery",
    Celery=lambda *a, **k: _FakeCelery(),
    shared_task=_noop_decorator,
)

# ---------------------------------------------------------------------------
# app.worker
# ---------------------------------------------------------------------------
_make_module("app.worker", celery=_FakeCelery())

# ---------------------------------------------------------------------------
# app.db.cassandra
# ---------------------------------------------------------------------------
_make_module("app.db")
_make_module("app.db.cassandra",
    get_session=lambda: None, close_session=lambda: None)

# ---------------------------------------------------------------------------
# app.db.dao.*
# ---------------------------------------------------------------------------
for _dao in ("app.db.dao.fusion_dao",
             "app.db.dao.opinion_dao",
             "app.db.dao.outcome_dao"):
    _make_module(_dao,
        fetch_opinions_st=lambda *a, **k: [],
        fetch_opinions_lt=lambda *a, **k: [],
        insert_opinion=lambda *a, **k: None,
        insert_claim_outcome=lambda *a, **k: None,
    )

# ---------------------------------------------------------------------------
# app.services.domain_decay
# ---------------------------------------------------------------------------
_make_module("app.services.domain_decay",
    get_domain_lambda=lambda domain: 0.08,
    auto_tune_lambda_task=_noop_decorator,
)

# ---------------------------------------------------------------------------
# app.services.expertise_temporal
# ---------------------------------------------------------------------------
_AgentExpertise = namedtuple(
    "AgentExpertise",
    ["calibration_score", "bias_score", "entropy_score", "prediction_count"],
)
_make_module("app.services.expertise_temporal",
    AgentExpertise=_AgentExpertise,
    fetch_temporal_expertise=lambda agent_id, domain:
        _AgentExpertise(0.8, 0.1, 0.7, 100),
)