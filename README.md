# fluss-monty

Check and run flows written as Python source in a [Monty](https://github.com/pydantic/monty)
sandbox. This is the core that the [fluss](https://github.com/arkitektio/fluss) client and the
fluss server share. It never talks to a server itself.

```python
from fluss_monty import ManifestAction, arun_python_flow, validate_python_flow

source = '''
async def main(image: MikroImage) -> MikroImage:
    return await segment(image, sigma=1.5)
'''

report = await validate_python_flow(source, [ManifestAction(alias="segment", action=segment_action)])
assert report.ok, report.diagnostics
report.args, report.returns, report.manifest, report.stubs  # what the flow is stored with

returns = await arun_python_flow(source, {"image": "img-1"}, actions, call)
```

- **Validation** works in two passes, and neither executes the entrypoint.
  1. The host `ast` checks the source's shape and reads the entrypoint's signature into ports.
  2. Monty type-checks the source against stubs of the manifest's actions, and nothing else. A
     call to anything outside the manifest is an unresolved name.

  `validate_python_flow_sync` is the same check for synchronous callers. Checking runs under
  tight resource limits (`CHECK_LIMITS`), because defining the source's functions runs their
  class bodies, decorators and defaults.
- **A run** binds only the manifest's actions, as `async` functions. Each call goes out through
  the injected `call(action, kwargs, call_key)`.
  - A call key is `alias:digest(kwargs):n`, the `n`-th identical call. It doesn't depend on how
    gathered calls interleave, so a workflow that is run again finds every call it made.
  - A run that ends early cancels the calls it still has in flight.
- **Structures** are their IDs. `@mikro/image` is typed `MikroImage`, a `str` subclass that only
  the type checker sees.

```bash
uv run pytest
```
