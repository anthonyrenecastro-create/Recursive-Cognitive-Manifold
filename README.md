# Recursive Cognitive Manifold

Recursive Cognitive Manifold is a Python prototype for experimenting with recursive manifold dynamics, region formation, semantic and episodic memory, topology repair, perturbation testing, persistence, and collapse/attractor diagnostics.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## CI-Friendly Test Commands

These commands are safe for headless CI runs and avoid polluting the repository with bytecode.

Run the unit test suite:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p 'test_*.py'
```

Run a syntax/import validation pass:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m compileall Foundation demo_driver.py rcm rcm_benchmarks.py rcm_collapse_integration.py rcm_persistence.py rcm_perturbation_testing.py regional_hrm_field.py
```

Run a headless smoke test of the demo pipeline:

```bash
PYTHONDONTWRITEBYTECODE=1 python demo_driver.py --output /tmp/rcm_demo.png
```

Run the lesion recovery experiment with visible collapse diagnostics:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m rcm.experiments.lesion_recovery
```

## Collapse Diagnostics Workflow

Collapse and attractor diagnostics are implemented in [rcm_collapse_integration.py](rcm_collapse_integration.py).

### Default Installation Paths

Collapse support is auto-installed in these entry surfaces:

- [Foundation](Foundation): extensionless compatibility loads patch `RecursiveCognitiveManifold` during import
- [rcm/experiments/specialization_demo.py](rcm/experiments/specialization_demo.py): direct package-driven demo runs install collapse support before execution

You can also install it manually in custom code:

```python
from rcm.cognition.manifold import RecursiveCognitiveManifold
from rcm_collapse_integration import install_collapse_attractor_support

install_collapse_attractor_support(RecursiveCognitiveManifold)
```

### Runtime Fields Added

After installation, manifolds expose:

- `collapse_flag`
- `collapse_energy`
- `collapse_memory`
- `attractor_anchor`
- `collapse_monitor`
- `last_collapse_metrics`

The wrapped `step()` result includes:

- `collapse`
- `region_collapse`
- `child_collapse`

The wrapped `snapshot()` includes:

- `collapse`
- `collapse_flag`
- `collapse_energy`
- `collapse_events`
- `attractor_initialized`

### Visible Diagnostics in Entry Points

The main demo path in [demo_driver.py](demo_driver.py) and [rcm/experiments/specialization_demo.py](rcm/experiments/specialization_demo.py):

- plots collapse energy in the rendered dashboard
- prints a collapse summary after execution
- includes collapse metrics in the final snapshot

The lesion recovery experiment in [rcm/experiments/lesion_recovery.py](rcm/experiments/lesion_recovery.py):

- prints a collapse summary after the post-lesion snapshot

### Test Coverage

Focused collapse tests live in [tests/test_collapse_integration.py](tests/test_collapse_integration.py) and cover:

- collapse flagging and event recording
- cooldown behavior
- attractor rebinding
- end-to-end wrapper behavior for `step()` and `snapshot()`
