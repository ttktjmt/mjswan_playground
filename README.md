# mjswan playground

<a href="https://github.com/ttktjmt/mjswan_playground/actions/workflows/deploy.yml"><img src="https://github.com/ttktjmt/mjswan_playground/actions/workflows/deploy.yml/badge.svg" alt="deploy"/></a>

A collection of tasks built with [mjswan](https://github.com/ttktjmt/mjswan).

## Tasks

| Task ID | Robot | Description | Preview & Link |
|---------|-------|-------------|----------------|
| [`husky`](src/mjswan_playground/husky/README.md) | Unitree G1 | Skateboarding under whole-body control ([HUSKY](https://husky-humanoid.github.io/), RSS 2026) | <a href="https://mjswan.com/s/oM-paPA"><img src="assets/husky.gif" width="200"/></a><br />[mjswan.com/s/oM-paPA](https://mjswan.com/s/oM-paPA) |
| [`wbc`](src/mjswan_playground/wbc/README.md) | Unitree G1 | One [wbc-mjlab](https://github.com/wbc-mjlab/wbc-mjlab) tracking policy over various motions | <a href="https://mjswan.com/s/XWwo7dV"><img src="assets/wbc.gif" width="200"/></a><br />[mjswan.com/s/XWwo7dV](https://mjswan.com/s/XWwo7dV) |
| [`pacman`](src/mjswan_playground/pacman/README.md) | Unitree G1 | Dodging thrown balls from a head depth camera ([PAC-MAN](https://lzyang2000.github.io/perceptive_cbf_rl/), 2026) | <a href="https://mjswan.com/s/GOTofiq"><img src="assets/pacman.gif" width="200"/></a><br />[mjswan.com/s/GOTofiq](https://mjswan.com/s/GOTofiq) |
| [`microduck`](src/mjswan_playground/microduck/README.md) | Microduck | Every policy the robot ships: walk, stand, sit, ground pick, ball kick, roulade, roller skate ([Microduck](https://github.com/pollen-robotics/microduck)) | <a href="https://mjswan.com/s/DOGILsh"><img src="assets/microduck.gif" width="200"/></a><br />[mjswan.com/s/DOGILsh](https://mjswan.com/s/DOGILsh) |
| [`musclemimic`](src/mjswan_playground/musclemimic/README.md) | MyoFullBody | A 354-muscle body tracking a walking clip with the public 2.05e9-step checkpoint ([MuscleMimic](https://github.com/amathislab/musclemimic), 2026) | <a href="https://mjswan.com/s/gOrSan8"><img src="assets/musclemimic.gif" width="200"/></a><br />[mjswan.com/s/gOrSan8](https://mjswan.com/s/gOrSan8) |
| [`spinkick`](src/mjswan_playground/spinkick/README.md) | Unitree G1 | A double spin kick motion tracking ([g1_spinkick_example](https://github.com/mujocolab/g1_spinkick_example)) | <a href="https://mjswan.com/s/-wXxa8L"><img src="assets/spinkick.gif" width="200"/></a><br />[mjswan.com/s/-wXxa8L](https://mjswan.com/s/-wXxa8L) |
| [`bipedhrl`](src/mjswan_playground/bipedhrl/README.md) | Unitree H1-2 | Walking to velocity commands with the checkpoint run on the real robot ([biped_hrl](https://github.com/spaethli/biped_hrl)) | <a href="https://mjswan.com/s/AacPm-T"><img src="assets/bipedhrl.gif" width="200"/></a><br />[mjswan.com/s/AacPm-T](https://mjswan.com/s/AacPm-T) |
| [`upkie`](src/mjswan_playground/upkie/README.md) | Upkie | A wheeled biped balancing and driving to velocity commands ([mjlab_upkie](https://github.com/MarcDcls/mjlab_upkie)) | <a href="https://mjswan.com/s/BIPHv2M"><img src="assets/upkie.gif" width="200"/></a><br />[mjswan.com/s/BIPHv2M](https://mjswan.com/s/BIPHv2M) |
| [`duet`](src/mjswan_playground/duet/README.md) | Unitree G1 | Walking and squatting down to a 0.18 m crouch while its arms move on their own ([DUET](https://github.com/bae-air-lab/DUET)) | <a href="https://mjswan.com/s/whAfOf-"><img src="assets/duet.gif" width="200"/></a><br />[mjswan.com/s/whAfOf-](https://mjswan.com/s/whAfOf-) |
| [`jumper`](src/mjswan_playground/jumper/README.md) | Jumper | A crab robot walking with a commanded body posture, walking on five legs, dancing, gesturing and jumping ([Jumper](https://github.com/KingKongRobotics/jumper)) | <a href="https://mjswan.com/s/7wT9SwQ"><img src="assets/jumper.gif" width="200"/></a><br />[mjswan.com/s/7wT9SwQ](https://mjswan.com/s/7wT9SwQ) |
| [`microduckpg`](src/mjswan_playground/microduckpg/README.md) | Microduck | Every experiment of a community playground: running, swing, basketball, stilts, desk and chimney climbs, long jump, backflip ([microduck-playground](https://github.com/Vottivott/microduck-playground)) | <a href="https://mjswan.com/s/_klQQmL"><img src="assets/microduckpg.gif" width="200"/></a><br />Moves: [mjswan.com/s/_klQQmL](https://mjswan.com/s/_klQQmL)<br />Parkour: [mjswan.com/s/mqwst_Y](https://mjswan.com/s/mqwst_Y)<br />Stilts: [mjswan.com/s/qK0Uwr7](https://mjswan.com/s/qK0Uwr7) |
| [`unitreerl`](src/mjswan_playground/unitreerl/README.md) | Unitree G1 | A G1 walking to velocity commands and dancing a LAFAN1 clip ([unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab)) | <a href="https://mjswan.com/s/opR2oo7"><img src="assets/unitreerl.gif" width="200"/></a><br />[mjswan.com/s/opR2oo7](https://mjswan.com/s/opR2oo7) |
| [`terra`](src/mjswan_playground/terra/README.md) | MyoFullBody | A 354-muscle body walking myosuite's terrain course, stairs, a beam and two jumps, and a flat route, with the TERRA-4B checkpoint ([TERRA](https://github.com/amathislab/terra), on [myosuite](https://github.com/MyoHub/myosuite)) | <img src="assets/terra.gif" width="200"/><br />WIP |
| [`wuji`](src/mjswan_playground/wuji/README.md) | Wuji Hand 1, Wuji Hand 2 | A dexterous hand turning a cube in its palm to random goal orientations, and spinning a pen along clips recorded on the real hand ([wuji-mjlab](https://github.com/wuji-technology/wuji-mjlab)) | <a href="https://mjswan.com/s/XbIngWj"><img src="assets/wuji.gif" width="200"/></a><br />Reorient: [mjswan.com/s/XbIngWj](https://mjswan.com/s/XbIngWj)<br />Penspin: [mjswan.com/s/JiGR41o](https://mjswan.com/s/JiGR41o) |
| [`asimov`](src/mjswan_playground/asimov/README.md) | Asimov 1 | Walking ([asimov1-locomotion-0818](https://huggingface.co/Menlo/asimov1-locomotion-0818)), and falling safely and getting up ([asimov-v1-policies](https://github.com/openhorizon-labs/asimov-v1-policies)) | <a href="https://mjswan.com/s/fePFavL"><img src="assets/asimov.gif" width="200"/></a><br />[mjswan.com/s/fePFavL](https://mjswan.com/s/fePFavL) |
| [`boosterdeploy`](src/mjswan_playground/boosterdeploy/README.md) | Booster K1 | Dancing and fighting along two clips with Booster Robotics' own BeyondMimic policies ([booster_deploy](https://github.com/BoosterRobotics/booster_deploy)) | <img src="assets/boosterdeploy.gif" width="200"/><br />WIP |

## CLI

```
uv run msp <subcommand>
```

| Subcommand | Description | Options |
|------------|-------------|---------|
| `list` | List the task IDs, then the parts | none |
| `run <task-id>` | Build a task and open it in the browser | `--host` (`localhost`), `--port` (`8080`), `--no-open`, `--output-dir` |
| `build <task-id>` | Build a task without launching | `--output-dir` |
| `site [<task-id>...]` | Build every task on the GitHub Pages site into one app | `--no-build`, `--base-path` (`/`), `--dist-dir` (`dist`), `--output-dir` (`dist/_site`) |

## Python

mjswan_playground can be imported as a Python package, and each task's builder can be used to build and launch the demo:

```python
import mjswan_playground

builder = mjswan_playground.load("husky")
builder.build(output_dir="dist").launch()
```

Each task also lives at `mjswan_playground.<task_module>.main` (the task ID with underscores, e.g. `mjswan_playground.husky.main`), exposing `setup_builder()`.

## License

This project is licensed under the [Apache-2.0 License](LICENSE).
The upstream assets each task fetches carry their own licenses. Please see each task's README and follow the respective license terms.
