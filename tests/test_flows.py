"""Python flows in a real Monty sandbox: validation, the port mapping, and runs.

Actions are plain async functions behind the injected ``call`` -- the seam the fluss client
fills with ``task.acall_raw`` -- so everything but the call itself is real.
"""

import ast
import asyncio
import time
from types import SimpleNamespace
from typing import Any

import pydantic_monty
import pytest
import pytest_asyncio

from fluss_monty import (
    ManifestAction,
    Port,
    PortKind,
    PythonFlowError,
    PythonFlowLimits,
    arun_python_flow,
    validate_python_flow,
    validate_python_flow_sync,
)
from fluss_monty.manifest import action_stub, build_stubs
from fluss_monty.ports import annotation_to_port, port_to_annotation, structure_class_name


def arg(key: str, kind: PortKind, **extra: Any) -> Port:
    return Port(key=key, kind=kind, nullable=extra.pop("nullable", False), **extra)


def ret(key: str, kind: PortKind, **extra: Any) -> Port:
    return Port(key=key, kind=kind, nullable=extra.pop("nullable", False), **extra)


IMAGE = "@mikro/image"


def action(hash: str, args: list, returns: list, kind: str = "FUNCTION") -> SimpleNamespace:
    return SimpleNamespace(hash=hash, kind=kind, args=args, returns=returns)


SEGMENT = ManifestAction(
    alias="segment",
    action=action("h-seg", [arg("image", PortKind.STRUCTURE, identifier=IMAGE), arg("sigma", PortKind.FLOAT, nullable=True)], [ret("return0", PortKind.STRUCTURE, identifier=IMAGE)]),
)
MEASURE = ManifestAction(
    alias="measure",
    action=action("h-measure", [arg("image", PortKind.STRUCTURE, identifier=IMAGE)], [ret("count", PortKind.INT), ret("area", PortKind.FLOAT)]),
)
MOVE = ManifestAction(alias="move_stage", action=action("h-move", [arg("x", PortKind.FLOAT)], []), effect="IRREVERSIBLE")
ACTIONS = [SEGMENT, MEASURE, MOVE]

SOURCE = '''"""Segment an image and count what is in it."""

import asyncio

THRESHOLD = 3


async def main(image: MikroImage, repeats: int = 1) -> tuple[MikroImage, int]:
    masks = await asyncio.gather(*[segment(image, sigma=1.5) for _ in range(repeats)])
    count, area = await measure(masks[0])
    print("counted", count, "over", area)
    return masks[0], count
'''


class Recorder:
    """The injected caller: records every call, answers from ``impls``."""

    def __init__(self, delay: float = 0.0) -> None:
        self.calls: list[tuple[str, dict, str]] = []
        self.delay = delay

    async def __call__(self, entry: ManifestAction, kwargs: dict, call_key: str) -> dict:
        self.calls.append((entry.alias, kwargs, call_key))
        await asyncio.sleep(self.delay)
        if entry.alias == "segment":
            if kwargs["image"] == "broken":
                raise RuntimeError("the segmenter crashed")
            return {"return0": f"{kwargs['image']}-mask"}
        if entry.alias == "measure":
            return {"count": 7, "area": 12.5}
        return {}


@pytest_asyncio.fixture
async def pool():
    async with pydantic_monty.AsyncMonty(min_processes=1, max_processes=2) as pool:
        yield pool


# --- ports <-> annotations ---------------------------------------------------------------


@pytest.mark.parametrize(
    "annotation",
    ["int", "float", "str", "bool", "MikroImage", "list[int]", "dict[str, MikroImage]", "list[MikroImage] | None", "dict[str, list[float | None]]"],
)
def test_annotations_and_ports_round_trip(annotation: str) -> None:
    structures = {"MikroImage": IMAGE}
    port = annotation_to_port(ast.parse(annotation, mode="eval").body, "x", structures)
    assert port_to_annotation(port) == annotation


def test_optional_spellings_are_nullable() -> None:
    for spelling in ("Optional[int]", "None | int", "'int | None'"):
        port = annotation_to_port(ast.parse(spelling, mode="eval").body, "x", {})
        assert (port.kind, port.nullable) == (PortKind.INT, True)


def test_structure_class_names() -> None:
    assert structure_class_name("@mikro/image") == "MikroImage"
    assert structure_class_name("@fluss/pythonflow") == "FlussPythonflow"


def test_optional_action_args_become_keyword_only() -> None:
    assert action_stub(SEGMENT) == "async def segment(image: MikroImage, *, sigma: float | None = ...) -> MikroImage: ..."
    assert action_stub(MEASURE) == "async def measure(image: MikroImage) -> tuple[int, float]: ..."
    assert action_stub(MOVE) == "async def move_stage(x: float) -> None: ..."


def test_stubs_declare_only_the_manifest() -> None:
    stubs, classes = build_stubs([SEGMENT], ["@mikro/roi"])
    assert classes == {"MikroImage": IMAGE, "MikroRoi": "@mikro/roi"}
    assert "measure" not in stubs


# --- validation --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_valid_flow_reports_its_ports_and_manifest(pool) -> None:
    report = await validate_python_flow(SOURCE, ACTIONS, pool=pool)
    assert report.ok, report.diagnostics
    assert [(p.key, p.kind, p.identifier, p.nullable, p.default) for p in report.args] == [
        ("image", PortKind.STRUCTURE, IMAGE, False, None),
        ("repeats", PortKind.INT, None, False, 1),
    ]
    assert [(p.key, p.kind) for p in report.returns] == [("return0", PortKind.STRUCTURE), ("return1", PortKind.INT)]
    assert [(m.alias, m.action_hash, m.effect) for m in report.manifest] == [("segment", "h-seg", "UNKNOWN"), ("measure", "h-measure", "UNKNOWN"), ("move_stage", "h-move", "IRREVERSIBLE")]
    assert report.runtime.startswith("monty-")


@pytest.mark.asyncio
async def test_an_optional_parameter_is_a_nullable_port(pool) -> None:
    report = await validate_python_flow("async def main(n: int | None = None) -> None:\n    pass\n", pool=pool)
    assert report.ok, report.diagnostics
    assert report.args[0].nullable and report.returns == []
    # The checker holds the signature to its annotation: `int = None` is not `int | None`.
    report = await validate_python_flow("async def main(n: int = None) -> None:\n    pass\n", pool=pool)
    assert not report.ok and "invalid-parameter-default" in report.diagnostics[0].message


@pytest.mark.parametrize(
    "source, message",
    [
        ("async def main(image: MikroImage) -> MikroImage:\n    return await delete_everything(image)\n", "delete_everything"),
        ("async def main(image: MikroImage) -> str:\n    return await measure(image)\n", "invalid-return-type"),
        ("async def main(x: int) -> int:\n    return await segment(x)\n", "invalid-argument-type"),
        ("print('side effect')\nasync def main() -> None:\n    pass\n", "module level"),
        ("def main() -> None:\n    pass\n", "async def"),
        ("async def other() -> None:\n    pass\n", "no `async def main"),
        ("async def main(x) -> None:\n    pass\n", "needs a type annotation"),
        ("async def main(x: set[int]) -> None:\n    pass\n", "has no port"),
        ("async def main(*xs: int) -> None:\n    pass\n", "named parameters only"),
        ("async def main( -> None:\n", "Syntax error"),
    ],
    ids=["unknown-name", "return-type", "argument-type", "top-level-code", "sync-main", "no-main", "unannotated", "unsupported-type", "varargs", "syntax"],
)
@pytest.mark.asyncio
async def test_invalid_flows_are_rejected(pool, source: str, message: str) -> None:
    report = await validate_python_flow(source, ACTIONS, pool=pool)
    assert not report.ok
    assert any(message in d.message for d in report.diagnostics), report.diagnostics


@pytest.mark.asyncio
async def test_generator_actions_and_duplicate_aliases_are_rejected(pool) -> None:
    stream = ManifestAction(alias="stream", action=action("h-gen", [], [ret("return0", PortKind.INT)], kind="GENERATOR"))
    report = await validate_python_flow("async def main() -> None:\n    pass\n", [stream], pool=pool)
    assert not report.ok and "FUNCTION" in report.diagnostics[0].message
    report = await validate_python_flow("async def main() -> None:\n    pass\n", [SEGMENT, SEGMENT], pool=pool)
    assert not report.ok and "unique" in report.diagnostics[0].message


# --- runs --------------------------------------------------------------------------------


async def _run(pool, source: str = SOURCE, kwargs: dict | None = None, caller: Recorder | None = None, **options: Any) -> tuple[dict, Recorder]:
    report = await validate_python_flow(source, ACTIONS, pool=pool)
    assert report.ok, report.diagnostics
    caller = caller or Recorder()
    returns = await arun_python_flow(source, kwargs or {}, ACTIONS, caller, args=report.args, returns=report.returns, pool=pool, **options)
    return returns, caller


@pytest.mark.asyncio
async def test_a_run_calls_its_actions_and_packs_its_returns(pool) -> None:
    printed: list[str] = []
    returns, caller = await _run(pool, kwargs={"image": "img-1", "repeats": 2}, print_callback=lambda stream, text: printed.append(text))
    assert returns == {"return0": "img-1-mask", "return1": 7}
    assert [(alias, kwargs) for alias, kwargs, _ in caller.calls] == [
        ("segment", {"image": "img-1", "sigma": 1.5}),
        ("segment", {"image": "img-1", "sigma": 1.5}),
        ("measure", {"image": "img-1-mask"}),
    ]
    assert len({call_key for _, _, call_key in caller.calls}) == 3
    assert "".join(printed) == "counted 7 over 12.5\n"


@pytest.mark.asyncio
async def test_gathered_calls_run_concurrently(pool) -> None:
    start = time.monotonic()
    await _run(pool, kwargs={"image": "img", "repeats": 5}, caller=Recorder(delay=0.3))
    # 5 segments + 1 measure at 0.3 s each: ~0.6 s concurrently, 1.8 s one by one.
    assert time.monotonic() - start < 1.4


@pytest.mark.parametrize(
    "source, kwargs, expected",
    [
        ("async def main() -> None:\n    await move_stage(1.0)\n", {}, {}),
        ("async def main(image: MikroImage) -> MikroImage:\n    return await segment(image)\n", {"image": "img"}, {"return0": "img-mask"}),
        ("async def main(image: MikroImage) -> tuple[int, float]:\n    return await measure(image)\n", {"image": "img"}, {"return0": 7, "return1": 12.5}),
    ],
    ids=["none", "one", "several"],
)
@pytest.mark.asyncio
async def test_returns_unwrap_and_pack(pool, source: str, kwargs: dict, expected: dict) -> None:
    returns, _ = await _run(pool, source, kwargs)
    assert returns == expected


@pytest.mark.asyncio
async def test_a_failing_action_fails_the_run(pool) -> None:
    with pytest.raises(PythonFlowError, match="the segmenter crashed") as caught:
        await _run(pool, kwargs={"image": "broken"})
    assert caught.value.traceback and "main" in caught.value.traceback


@pytest.mark.asyncio
async def test_a_flow_can_catch_a_failing_action(pool) -> None:
    source = (
        "async def main(image: MikroImage) -> str:\n"
        "    try:\n"
        "        return await segment(image)\n"
        "    except RuntimeError as error:\n"
        "        return 'recovered: ' + str(error)\n"
    )
    returns, _ = await _run(pool, source, {"image": "broken"})
    assert returns == {"return0": "recovered: segment failed: the segmenter crashed"}


@pytest.mark.asyncio
async def test_the_suspension_limit_stops_a_runaway_flow(pool) -> None:
    source = "import asyncio\n\nasync def main() -> None:\n    await asyncio.gather(*[move_stage(float(i)) for i in range(50)])\n"
    with pytest.raises(PythonFlowError, match="suspension limit"):
        await _run(pool, source, limits=PythonFlowLimits(max_suspensions=10))


@pytest.mark.asyncio
async def test_a_run_rechecks_the_source_against_the_manifest_it_is_given(pool) -> None:
    """A run binds only what it is given now: a source calling an action outside it never starts."""
    caller = Recorder()
    with pytest.raises(PythonFlowError, match="does not check"):
        await arun_python_flow(SOURCE, {"image": "img"}, [SEGMENT], caller, pool=pool)
    assert caller.calls == []


@pytest.mark.asyncio
async def test_the_sandbox_has_no_host_filesystem(pool) -> None:
    """`open` is not even a name a flow can check against; nor does a run bind any OS access."""
    report = await validate_python_flow("async def main() -> str:\n    return open('/etc/hostname').read()\n", ACTIONS, pool=pool)
    assert not report.ok and "`open`" in report.diagnostics[0].message
    report = await validate_python_flow("import os\n\nasync def main() -> list[str]:\n    return os.listdir('/')\n", ACTIONS, pool=pool)
    assert report.ok, report.diagnostics
    with pytest.raises(PythonFlowError, match="PermissionError"):
        await arun_python_flow("import os\n\nasync def main() -> list[str]:\n    return os.listdir('/')\n", {}, ACTIONS, Recorder(), returns=report.returns, pool=pool)


# --- resuming: call keys and a fixed sandbox ----------------------------------------------

CHAINS = """import asyncio


async def chain(image: MikroImage) -> MikroImage:
    mask = await segment(image)
    return await segment(mask)


async def main(a: MikroImage, b: MikroImage) -> None:
    await asyncio.gather(chain(a), chain(b))
"""


class SlowFor(Recorder):
    """Answers calls on ``slow`` images late, so the other chain's calls go out first."""

    def __init__(self, slow: str) -> None:
        super().__init__()
        self.slow = slow

    async def __call__(self, entry: ManifestAction, kwargs: dict, call_key: str) -> dict:
        self.calls.append((entry.alias, kwargs, call_key))
        await asyncio.sleep(0.2 if kwargs["image"].startswith(self.slow) else 0.0)
        return {"return0": f"{kwargs['image']}-mask"}


@pytest.mark.asyncio
async def test_a_call_keeps_its_key_however_its_siblings_interleave(pool) -> None:
    """A resumed run's calls arrive in another order; each must still find its own result."""
    keyed = []
    for slow in ("a", "b"):
        _, caller = await _run(pool, CHAINS, {"a": "a", "b": "b"}, caller=SlowFor(slow))
        keyed.append({call_key: kwargs["image"] for _, kwargs, call_key in caller.calls})
        order = [kwargs["image"] for _, kwargs, _ in caller.calls]
        assert order[2] == ("b-mask" if slow == "a" else "a-mask"), order
    assert keyed[0] == keyed[1]


@pytest.mark.asyncio
async def test_a_call_key_names_the_alias_its_arguments_and_their_repeat(pool) -> None:
    source = "async def main(image: MikroImage) -> None:\n    await segment(image)\n    await segment(image)\n    await segment(image, sigma=2.0)\n"
    _, caller = await _run(pool, source, {"image": "img"})
    keys = [call_key for _, _, call_key in caller.calls]
    assert [key.split(":")[0] for key in keys] == ["segment"] * 3
    assert [key.split(":")[2] for key in keys] == ["1", "2", "1"]
    assert keys[0].split(":")[1] == keys[1].split(":")[1] != keys[2].split(":")[1]


@pytest.mark.asyncio
async def test_a_fixed_seed_makes_the_sandbox_draw_the_same_numbers(pool) -> None:
    source = "import random\n\nasync def main() -> float:\n    return random.random()\n"
    policy = {"random_start": {"seed": "run-1:task-1"}}
    first, _ = await _run(pool, source, os_policy=policy)
    second, _ = await _run(pool, source, os_policy=policy)
    other, _ = await _run(pool, source, os_policy={"random_start": {"seed": "run-2:task-2"}})
    assert first == second != other


@pytest.mark.asyncio
async def test_stored_ports_that_disagree_with_the_source_fail_the_run(pool) -> None:
    caller = Recorder()
    wrong = [ret("count", PortKind.INT)]
    with pytest.raises(PythonFlowError, match="stored returns"):
        await arun_python_flow(SOURCE, {"image": "img"}, ACTIONS, caller, returns=wrong, pool=pool)
    assert caller.calls == []


@pytest.mark.asyncio
async def test_an_alias_may_not_shadow_the_entrypoint_or_a_structure(pool) -> None:
    for alias in ("main", "asyncio", "MikroImage"):
        shadow = ManifestAction(alias=alias, action=SEGMENT.action)
        report = await validate_python_flow(SOURCE, [shadow], pool=pool)
        assert not report.ok and "shadow" in report.diagnostics[0].message, alias




def test_a_sync_caller_gets_the_same_report() -> None:
    """The server validates in a synchronous mutation."""
    report = validate_python_flow_sync(SOURCE, ACTIONS)
    assert report.ok, report.diagnostics
    assert [(port.key, port.kind) for port in report.returns] == [("return0", PortKind.STRUCTURE), ("return1", PortKind.INT)]
    broken = validate_python_flow_sync(SOURCE.replace("measure(", "count("), ACTIONS)
    assert not broken.ok and "count" in broken.diagnostics[0].message


@pytest.mark.asyncio
async def test_checking_a_source_is_bounded(pool) -> None:
    """Defining the source evaluates its defaults: on a server, someone else's code."""
    source = "def spin(x: int = sum(range(10**12))) -> int:\n    return x\n\n\nasync def main() -> None:\n    return None\n"
    report = await validate_python_flow(source, ACTIONS, pool=pool, limits={"max_feed_duration_secs": 0.3})
    assert not report.ok and "time limit" in report.diagnostics[0].message


def test_each_type_error_is_a_diagnostic_on_its_line() -> None:
    source = "async def main(image: MikroImage) -> int:\n    mask = await segment(3)\n    return await measure(mask, extra=1)\n"
    report = validate_python_flow_sync(source, ACTIONS)
    assert not report.ok
    assert sorted({d.line for d in report.diagnostics}) == [2, 3]
    assert "segment" in report.diagnostics[0].message and "flow.py" not in report.diagnostics[0].message


def test_a_definition_that_raises_is_a_diagnostic_on_its_line() -> None:
    source = "def helper(x: int = 1 // 0) -> int:\n    return x\n\n\nasync def main() -> None:\n    return None\n"
    report = validate_python_flow_sync(source, ACTIONS)
    assert not report.ok
    assert report.diagnostics[0].line == 1 and "ZeroDivisionError" in report.diagnostics[0].message
