# Installation

`t4perceval` needs **Python 3.10 or newer**. The evaluation core depends only on NumPy, SciPy,
PyArrow, Shapely and attrs -- reading a dataset is an optional extra, not part of the base install.

## Install the core

=== "uv"

    ```console
    uv add t4perceval
    ```

=== "pip"

    ```console
    pip install t4perceval
    ```

The package is not on PyPI yet, so until the first release install it from the repository:

```console
pip install 'git+https://github.com/ktro2828/t4perceval'
```

## Extras

Importers are optional. The evaluation core reads and writes its own
[recording model](../concepts/data-model.md) and never imports either dependency, and keeping them
out of the base install is what mechanically enforces that rule: an accidental `import t4_devkit`
outside `t4perceval.importer` fails for anyone who installed without the extra.

| Extra    | Pulls in                                        | Needed for                                                                 |
| :------- | :---------------------------------------------- | :------------------------------------------------------------------------- |
| `t4`     | `t4-devkit`                                     | [Reading a T4 dataset](../user-guide/dataset-importers.md#t4-dataset)      |
| `rosbag` | `mcap`, `mcap-ros2-support`, `zstandard`, `lz4` | [Reading an MCAP ROS bag](../user-guide/dataset-importers.md#mcap-ros-bag) |

```console
pip install 't4perceval[t4]'
pip install 't4perceval[rosbag]'
pip install 't4perceval[t4,rosbag]'
```

An MCAP bag is decoded from the message definitions it embeds, so the `rosbag` extra needs **no ROS
installation** and no Autoware message packages.

## Verify the install

```console
python -c "import t4perceval; print(t4perceval.Store())"
```

## Set up for development

```console
git clone https://github.com/ktro2828/t4perceval
cd t4perceval
uv sync --group dev
uv run pytest tests -q
```

`uv sync --group dev` installs the test runner, the documentation toolchain and `t4-devkit`. Add
`--all-extras` if you also want the ROS bag importer's dependencies:

```console
uv sync --group dev --all-extras
```

See [Contributing](../development/contributing.md) for the lint and documentation commands.

## Next steps

- [Quick start](quickstart.md) -- the five things `t4perceval` is made of.
- [First evaluation](first-evaluation.md) -- ground truth in, mAP out.
