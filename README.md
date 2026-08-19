# mjswan playground

A collection of tasks built with [mjswan](https://github.com/ttktjmt/mjswan).

## Tasks

| Task ID | Robot | Description | Link |
|---------|-------|-------------|------|
| [`husky-skater`](src/mjswan_playground/husky_skater/README.md) | Unitree G1 | Skateboarding under whole-body control ([HUSKY](https://husky-humanoid.github.io/), RSS 2026) | [WIP] |
| [`wbc-g1`](src/mjswan_playground/wbc_g1/README.md) | Unitree G1 | One [wbc-mjlab](https://github.com/wbc-mjlab/wbc-mjlab) tracking policy over various motions | [WIP] |

## CLI

```
uv run mjswan-playground <subcommand>
```

| Subcommand | Description | Options |
|------------|-------------|---------|
| `list` | List the task IDs | — |
| `run <task-id>` | Build a task and open it in the browser | `--host` (`localhost`), `--port` (`8080`), `--no-open`, `--output-dir` |
| `build <task-id>` | Build a task without launching | `--output-dir` |

## Python package

mjswan_playground can be imported as a Python package, and each task's builder can be used to build and launch the demo:

```python
import mjswan_playground

builder = mjswan_playground.load("husky-skater")
builder.build(output_dir="dist").launch()
```

Each task also lives at `mjswan_playground.<task_module>.main` (the task ID with underscores, e.g. `mjswan_playground.husky_skater.main`), exposing `setup_builder()`.

## License

This project is licensed under the [Apache-2.0 License](LICENSE).
The upstream assets each task fetches carry their own licenses. Please see each task's README and follow the respective license terms.
