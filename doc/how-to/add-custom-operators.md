# Add Custom Operators

This guide shows you how to add your own transform to contract `apply:` pipelines. Operator semantics and tiers: [operators reference](../reference/contract.md#operators).

## Register a built-in

Add the operator to `rosetta/contract/builtin_operators.py`. The framework (`rosetta/contract/operators.py`) and the contract schema do not change:

```python
from rosetta.contract.operators import register_operator, Operator, Invertibility

@register_operator("my_op", kind=Invertibility.BIJECTIVE)
class MyOperator(Operator):
    def forward(self, arr): ...
    def inverse(self, arr): ...   # round-trip verified at contract load
```

Registration enforces the tier's promises up front. A serveable tier (`BIDIRECTIONAL`/`BIJECTIVE`) without an `inverse` is rejected at import. A name already registered raises unless you pass `override=True` (same rule as the codec registries), so a plugin never silently shadows a built-in like `clamp`. An operator fixing image geometry declares the output by setting `self.output_hw = (h, w)`. Image observations require some operator in their pipeline to declare geometry (built-in `resize` does).

## Register from a plugin

No fork needed. Package your operator and advertise the module under the `rosetta.operators` entry-point group. Rosetta discovers and imports the module at contract load, so the contract references the operator by name only (no module paths in the YAML):

```toml
# in your plugin's pyproject.toml
[project.entry-points."rosetta.operators"]
my_operators = "my_pkg.my_operators"   # imported once; its @register_operator calls run
```

```yaml
apply: [my_op]             # resolves to the plugin's registered operator
```

## Pick the right tier

- `FORWARD_ONLY`: lossy, decode/build only. Rejected on actions at load.
- `BIDIRECTIONAL`: runs both ways but lossy (e.g. a bound). Allowed on actions.
- `BIJECTIVE`: inverse exactly undoes forward. Round-trip verified at load.

A wrong inverse fails at load instead of corrupting actions silently.
