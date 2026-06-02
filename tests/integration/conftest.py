"""Keep the live-service integration runner out of normal pytest collection.

`run_e2e.py` requires a running Qdrant + Redis and uses ordered, stateful setup;
it is invoked explicitly via `python -m tests.integration.run_e2e`, not pytest.
"""

collect_ignore = ["run_e2e.py", "helpers.py"]
