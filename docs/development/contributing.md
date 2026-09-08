# Contributing

## Setting up

```console
git clone https://github.com/ktro2828/t4perceval
cd t4perceval
uv sync --group dev --all-extras
```

`--group dev` brings the test runner, the documentation toolchain and `t4-devkit`; `--all-extras`
adds the ROS bag importer's dependencies. Drop `--all-extras` if you are not touching that importer.

## The checks

```console
uv run pytest tests -q
uv run ruff check t4perceval tests
uv run ruff format --check t4perceval tests
```

Pre-commit runs the lint and format checks plus the Markdown linter:

```console
uv run pre-commit install
uv run pre-commit run --all-files
```

## Documentation

```console
uv run zensical serve      # live preview
uv run zensical build --clean
```

For documentation-only work the `docs` group is enough, and installs 25 packages instead of the dev
group's 69:

```console
uv sync --only-group docs --no-install-project
uv run --no-sync zensical serve
```

That works because mkdocstrings renders the [API reference](../reference/api/index.md) through
griffe, which parses the source **statically** -- the package does not have to be importable. A
signature or docstring change therefore updates the reference automatically. Everything else is
written by hand.

`.github/workflows/docs.yml` builds with exactly those two commands and publishes `site/` to GitHub
Pages on every push to `main`, so a docs change is live once it merges. It can also be re-run by
hand from the Actions tab.

When you add a page, add it to the `nav` in `zensical.toml`. The navigation is organised by **user
intent**, not by the Python package hierarchy -- see [the index](../index.md#how-this-documentation-is-organised)
for what belongs where.

## What the code expects of a change

The [design principles](design-principles.md) are the short version. In practice:

- **Docstrings carry the reasoning.** They are the API reference, and the ones in this codebase
  explain _why_ a thing is shaped as it is, not just what it does. Match that.
- **A new component gets a descriptor named for meaning**, not for the archetype using it. See
  [Extending components](extending-components.md).
- **A new archetype composes, it does not inherit.** See
  [Extending archetypes](extending-archetypes.md).
- **A new system declares `REQUIRES` and `PROVIDES` honestly**, and writes only its declared
  targets. See [Extending systems](extending-systems.md).
- **A silent-wrong becomes an exception** where it can. This domain's failures are plausible
  numbers, not crashes.
- **Comments explain the non-obvious decision**, not the mechanics. The existing comments are a good
  guide to the expected density.

## Tests

- Tests live in `tests/`, one file per area, with classes grouping behaviours.
- Importers are tested against **synthetic fixtures built in-process** (`tests/t4_builder.py`,
  `tests/rosbag_builder.py`) rather than against a checked-in dataset.
- `tests/test_importer_isolation.py` asserts the dependency direction: importing `t4perceval` must
  not pull in `t4_devkit` or the MCAP libraries. Do not break it.
- A behaviour that would silently produce a wrong number deserves a test that it raises.

## Benchmarks

```console
uv run python benchmarks/compare.py --check
```

`--check` fails on any numerical difference from `perception_eval` that is not already classified in
[Metric divergences](metric-divergences.md). If your change moves a number, either fix it or
document the divergence there -- and regenerate `benchmarks/results/latest.md`. See
[Benchmarks](benchmarks.md).

## Ruff

The rule set is **pinned** in `pyproject.toml` (`select = ["E4", "E7", "E9", "F"]`) rather than
inherited from whatever Ruff enables by default this month. Widening it is a deliberate change made
there, with the reason recorded, not a side effect of an upgrade.

`from __future__ import annotations` is required at the top of every module.

## Commits and pull requests

- One concern per commit.
- If a change alters a documented behaviour, update the page in the same commit.
- If it changes an architectural decision, add or supersede an
  [ADR](design-decisions/index.md).

## Where to go next

- [Roadmap](roadmap.md) -- what is wanted.
- [Architecture](architecture.md) -- where things live.
