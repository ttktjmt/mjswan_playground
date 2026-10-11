---
name: add-new-task
description: Add a task to mjswan_playground from a repo that registers mjlab tasks, given as a GitHub URL or a local path, with every simulation in it that mjswan can run as a scene. Ports each with mjswan's mjlab-to-mjswan skill, lands them as src/mjswan_playground/<task-id>/ wired into the registry, the msp CLI and the README, reviews the diff with mjswan's simplify-comments, and opens one pull request for the task. Use when the user wants a new task or demo in this playground.
argument-hint: <repo-url-or-path> [task-id]
---

Add one task to this playground, on a branch and a pull request of its own: one upstream repository, with every simulation in it that mjswan can run, each a scene of the task's project (`jumper`, `microduckpg`). A simulation is an mjlab task the repository registers, or an experiment it ships, with a trained policy of its own. The arguments are the target repo and, optionally, the playground id. The port itself is mjswan's `mjlab-to-mjswan` skill: this file says where the port lands, what else a playground task needs, and how it ends. Where the two disagree, this file wins. Quoted names are that skill's section headings.

| mjlab-to-mjswan | Here |
|---|---|
| "Ground rules" | as written, plus the four below |
| "1. Acquire the target" | replaced by step 2 |
| "2. Find the task ids", "3. Pre-flight" | as written, with step 3's notes |
| "4. Get the policy into ONNX" | replaced by step 4 |
| "5. Generate the port" | replaced by step 5 |
| "6. Build, then fix what fails", "7. Parity gate" | as written, with step 6's commands and notes |
| "8. When the gap is mjswan's, open a PR" | as written, plus step 9 |
| "9. Report" | replaced by steps 10 and 11 |

- Nothing binary goes into git but the preview GIF, and nothing is cloned into the tracked tree: upstream code, checkpoints and clips are fetched at build time from pinned sources into `.cache/`.
- Unattended mode, when the caller says so (a routine prompt does): ask nothing. Every answer comes from what the caller handed over, and a question it does not answer is a stop.
- Each simulation passes or fails on its own. One that hits a stop is left out of the task, and the pull request names it with the step, the error verbatim and what would unblock it; the rest go on.
- A stop that leaves no simulation, in either mode: no commit, no push, no pull request. Report the step, the error verbatim and what would unblock it.

## 0. Load mjlab-to-mjswan

Read the copy of the skill that matches the mjswan this repo locks, even when a plugin install carries another: the instructions and the installed API then agree. That is the release tag, or the commit a step 9 pin names:

```sh
SRC=$(grep -A2 '^name = "mjswan"$' uv.lock)
REF=$(echo "$SRC" | grep -o '#[0-9a-f]*' | tr -d '#')
[ -n "$REF" ] || REF=v$(echo "$SRC" | sed -n 's/^version = "\(.*\)"$/\1/p')
MJSWAN=.cache/mjswan-$REF
if [ ! -d "$MJSWAN" ]; then
  git init -q "$MJSWAN"
  git -C "$MJSWAN" fetch -q --depth 1 https://github.com/ttktjmt/mjswan "$REF"
  git -C "$MJSWAN" -c advice.detachedHead=false checkout -q FETCH_HEAD
fi
```

Then follow `$MJSWAN/skills/mjlab-to-mjswan/SKILL.md`. Its `export_policy.py` sits beside it, and step 10 uses `$MJSWAN/.claude/commands/simplify-comments.md`.

## 1. Intake: ask here, not later

- The target: a GitHub URL or a local path. A local path also needs its URL, since the task fetches upstream at build time.
- The simulations: every mjlab task upstream registers, and every experiment it ships, with a trained policy of its own (step 3). A list the caller hands over, such as a backlog entry's `simulations`, is where to start, not where to stop.
- The playground id, as short and plain as the ones already here (`husky`, `wbc`, `pacman`, `microduck`, `musclemimic`): the project's own short name as one lowercase word, without the robot, `example`, a version or separators unless they are what sets it apart (`spinkick`, not `g1_spinkick_example`). It matches `[a-z][a-z0-9_]*`, is new to `mjswan_playground.registry.ALL_TASKS`, and becomes the module name.
- The license of everything the build fetches, for every simulation: code, checkpoints, clips, meshes. Clips derived from a dataset carry the dataset's terms even inside an Apache-2.0 repo, and a repo that states none means checking where they came from. None, or NC / SA / ND / GPL: say so now. The task can still land, with a License section in its README, but only its author publishes it (see `husky`, `microduck`). Unattended, a license the caller's answer does not cover is a stop.
- One repository, one task, one branch, one pull request. `git fetch origin main`, then use the branch the caller names, else the current branch if it has no commits beyond `origin/main`, else `claude/add-<id>`; a new branch starts from `origin/main`. Unrelated changes in the working tree are a stop.

## 2. Pin, then pick the shape (replaces "Acquire the target")

- Every repo the build reads files from is pinned to a commit (`git ls-remote <url> HEAD`) and fetched through `mjswan_playground._deps.ensure_repo(name=..., url=..., commit=..., marker=..., root_env_var="MJSWAN_<NAME>_ROOT")`, which checks out into `.cache/`. A local path is only read: point that variable at it.
- The interpreter is the playground's (`uv sync`). Never install into the target.
- Take the first shape that fits. [reference/shapes.md](reference/shapes.md) walks through the five tasks as precedents; read the precedent's files before writing.

  | Shape | When | Registration | Precedent |
  |---|---|---|---|
  | extra | upstream is a package the playground can resolve | `[project.optional-dependencies] <id>`, commented only for what its name does not say, from PyPI (`uv.lock` is then the pin; check the release matches the commit you read) or from a git source under `[tool.uv.sources]` pinned with `rev = "<commit>"`, the same commit as any `ensure_repo` checkout of that repo; `uv lock` and `uv sync --extra <id>` before step 3 | `wbc`, `musclemimic` |
  | checkout | upstream runs only from source, on imports the playground already resolves | `upstream.py` with `resolve_root()` and `register_tasks(root)`, the root first on `sys.path`; version shims marked `# ponytail:` | `pacman` |
  | data-only | the env config cannot be adapted: a non-mjlab actuator, pins that do not co-resolve | follow the task here that already runs the same robot this way (`microduck` for a Microduck repo, whose env configs drive BAM); with none, out of scope: stop and propose the `husky` / `microduck` shape | `husky`, `microduck` |

- The playground pins `mjlab==1.6.0`. A resolver conflict on `mujoco` or `mjlab` is mjlab-to-mjswan's stop: report the resolver output verbatim. When upstream locks another mjlab but resolves anyway, diff its task config and robot constants between the two versions and list what really differs under "What differs from upstream"; shims are only for what fails to import.

## 3. Find the simulations, pre-flight each

"2. Find the task ids" and "3. Pre-flight, seconds before any build" as written, run with `uv run python` and importing through the registration above, with these notes:

- An upstream that registers through mjlab's `mjlab.tasks` entry point is loaded by `import mjlab` itself, so the registry diff comes out empty. Read its mjlab task ids from its `register_mjlab_task` calls, or diff with its skip switch set if it has one.
- List every mjlab task upstream registers and every experiment it ships, each with its published checkpoint or that it has none. Each one with a checkpoint is a simulation to port, and pre-flight runs on each; one without is left out. Ask for env vars and credentials here; unattended, a simulation that needs one the caller did not hand over is left out.
- An action term pre-flight rejects is sorted the way "8. When the gap is mjswan's" sorts it: a generic one goes to step 9, a task-specific one is the data-only stop for that simulation.

## 4. Policies and clips (replaces "Get the policy into ONNX")

Every checkpoint is fetched at build time from a pinned source:

- a git repo, upstream's or a deploy repo: the `ensure_repo` checkout;
- the Hub: `add_policy_hf(repo_id, revision=<commit>)`;
- W&B: `add_policy_wandb(run_path)`, for a public run only;
- a `.pt` in a pinned source: converted once at build time into `CACHE_DIR / "<id>"`, the way `musclemimic/upstream.py` caches its conversion, with the calls `export_policy.py` makes (`create_pt_onnx_export_context`, `align_obs_normalizer`);
- a `.pt` only on someone's disk: stop. It has to be published first; the Hub is the short path.

When upstream publishes more than one checkpoint of one policy, prefer the one that ships with a deploy contract and clips, and name the others in the pull request. Policies for different behaviours of one task, such as `jumper`'s dances and gestures, are each ported, on that task's scene.

`policy_joint_names` and `default_joint_pos` come from upstream's deploy contract when it ships one (`wbc`: `config.yaml`, `pacman`: `g1_deploy_constants.py`, `duet`: `deploy_contract.json`), else from an mjlab export's own metadata (`mjswan.mjlab.onnx_meta.read_mjlab_metadata`, rounded to three decimals). They name the joints the policy's actions drive, in output order, since their count sets the action count: a contract that also lists every observed joint (`duet`'s `joint_names` beside `action_joint_names`) gives its action list and that list's offsets. `encoder_bias` comes from the local `.pt` path's `policy_meta.json` or a contract that carries one; otherwise leave it unset and say so in the README.

A tracking task also needs its clips, pinned and fetched like the checkpoints: one `add_motion` per clip, with `anchor_body_name` and `body_names` from the task's motion command, and the clip list and default from upstream's manifest when it ships one. A clip published only to W&B is usually inside the checkpoint anyway: an mjlab tracking export bakes it into the ONNX, where a second input, `time_step`, selects the frame and the outputs after the action carry that frame's reference for the tracked bodies only. Rebuild a full mjlab motion `.npz` from those tables at build time into `.cache/<id>/`, replaying each frame the way `mjlab.scripts.csv_to_npz` does, and refuse a rebuild whose tracked bodies disagree with the baked ones. That policy takes `in_keys=[DEFAULT_OBS_GROUP_KEY, "time_step"]` and `out_keys` naming every output, `"action"` first.

## 5. Generate into the package (replaces "Generate the port")

```
src/mjswan_playground/<id>/
  __init__.py   a one-line docstring
  main.py       pinned constants, then setup_builder() -> mjswan.Builder
  terms.py      only terms that failed to trace or that msp build refuses; mjswan's rules for it stand
  upstream.py   the checkout shape, or fetching and converting past a few lines
  README.md     from reference/task-readme.md
```

- `main.py` exposes `setup_builder()` and runs nothing: no `main()`, `build()` or `launch()`. `msp run` and `msp build` do that. With an extra, it imports upstream's package by name, so a missing extra fails as an import error rather than a bare `KeyError` from `load_env_cfg`.
- One project for the repository, with a scene for each simulation in upstream's order: one `add_scene_mjlab` per mjlab task, with its policies on it (`jumper/main.py`). Once `main.py` grows past a screen, each simulation gets a module beside it (`jumper`, `microduckpg`).
- Register it now, since `msp build` looks it up: `"<id>": "mjswan_playground.<id>.main"` in `registry.py`, in alphabetical order.
- The README follows [reference/task-readme.md](reference/task-readme.md).
- The project carries upstream's license file verbatim: `builder.add_project(name=..., license=<checkout> / "LICENSE")`, and `project.set_notice(<checkout> / "NOTICE")` beside it when upstream ships one. A checkpoint from elsewhere whose terms you could not read leaves the project without one, and the pull request says so.
- A shim for an upstream version gap carries `# ponytail: <what>; drop once <condition>.` (pacman's precedent). A gap in mjswan itself gets no shim: step 9.

## 6. Build, inspect, parity

"6. Build, then fix what fails" and "7. Parity gate" as written, with these commands:

```sh
uv run msp build <id>             # writes dist/<id>/
uv run mjswan info dist/<id>      # motions are not listed: read dist/<id>/manifest.json
```

- A term that will not trace but does nothing in the play config (disabled, or zero ranges) is dropped from `env_cfg` and listed under "What differs from upstream", not sent to step 9. Name the mjswan gap behind it in the pull request.
- Parity is `MUJOCO_GL=disable uv run --with onnxruntime python scripts/parity.py <id>`, under `--with onnxruntime` since the playground's own environment may lack it. It checks every scene's env as `main.py` builds it, taking each `env_cfg` that `setup_builder()` passes to `add_scene_mjlab` and importing `mjswan_playground.<id>.terms` if it exists. A tracking task whose `env_cfg` leaves the clip unset takes `--motion-file <clip the build bundles>`, since the env loads it on construction. Any task's parity reruns the same way.
- `run_parity` in mjswan 0.11.1 marks a termination that is not native `OK ... over 0 steps` without comparing it. Report those terminations as unchecked by parity.
- A lag the policy trained with is part of its input. An observation term whose `delay_min_lag` and `delay_max_lag` are both `k` reads the value `k` control steps back; mjswan drops those fields, so give the term `history_steps=(k,)`. mjswan does not model an actuator command delay either: in a data-only scene, put a first-order filter on each actuator instead (`dyntype` `filterexact`, `dynprm[0]` the time constant, the keyframe's `act` at its `ctrl`), and let the preview settle its length.
- `run_parity` checks no command, so the script also runs `mjswan.compile.run_command_parity` on each traced command, built as the Builder builds it. It compares the graph with the term it traced, so a command `terms.py` rewrites is also checked against upstream's term in mjlab.
- `msp build` fails on a graph input that reads a command field the browser does not serve, and names it with the fields that command does serve. The browser would never run that graph, while parity still passes. Copy the term into `terms.py` if it is upstream's, and read one of those fields instead.

## 7. Wire it in

- `README.md`: a row at the end of the Tasks table linking `src/mjswan_playground/<id>/README.md`, with the robot, a one-line description like the others that names the simulations when there are several, and `<img src="assets/<id>.gif" width="200"/><br />WIP` as its Preview & Link cell. Whoever publishes the task replaces `WIP` with the link. The GitHub Pages site lists the tasks in this table's order, and its build refuses a task without a row. A task whose build needs a login, such as a gated Hub download, keeps its row and goes in `NOT_ON_SITE` in `src/mjswan_playground/_site.py` with the reason, since the deploy workflow has none.
- The preview: a `PREVIEWS` entry in `scripts/record_preview.py` with a cut for each scene (`Preview(scene=...)`, as `microduckpg` has), framed with `--shot` starting from the closest precedent's, then `assets/<id>.gif` filmed with it. Without a GPU (a cloud session), film with `xvfb-run -a uv run --group previews python scripts/record_preview.py <id> --software --chromium /opt/pw-browsers/chromium`. A preview that will not film at all is not a stop: the cell keeps only `WIP`, and the pull request says why.
- Filming checks the task as it runs. The app runs on for 20 s after the clip, and the recording fails, exit 1, on a termination other than `time_out`, a root tipped past 75°, NaN, an observation input that stays all zeros, a traced graph that stops running, actions that never change, a stalled step loop, a page error, or a slow-motion or choppy clip. It writes `dist/preview/<id>.json` and a contact sheet, `dist/preview/<id>.png`: every frame of the GIF, then the run after the clip, failures outlined in red. Read the sheet: the Read tool shows a GIF's first frame only.
- Loop until a round is clean: no failure, and nothing on the sheet that the command or clip does not call for (a fall, a snap back to the start pose, a robot that freezes, jitters or slides). Each round takes the report's first failure and its time, finds the cause with step 6's tools (parity on the term behind it, its inputs in `manifest.json`), fixes the port, rebuilds with `msp build <id>` and films again. A cause in mjswan itself goes to step 9, and the rounds go on against its pin. A slow or choppy clip is the recording's, not the port's: film with `--software`. Fix the cause, never the check: `upright=False` and `ok_terminations` are for what the task itself plays out, a flip or a dodgeball hit, and the pull request says which and why. A scene with no clean cut after five rounds leaves its simulation out, with each round's failures and what it changed.

## 8. Verify

`make format` and `make test` (the registry and README tests pick up the new id). Then delete this task's checkouts and conversions under `.cache/`, and its Hub downloads under `~/.cache/huggingface/hub/` (`hf_hub_download` caches there), and build once more: the build has to fetch everything it needs by itself.

## 9. When the gap is mjswan's

"8. When the gap is mjswan's, open a PR" as written, with these changes:

- Ask before opening anything; unattended, the caller's go-ahead covers it. Reuse an open ttktjmt/mjswan PR labelled `from-playground` for the same term instead of opening a second one.
- Work in a clone at `.cache/mjswan-pr`. Push the branch to ttktjmt/mjswan itself when you can (a cloud session needs it attached with push access), else fork as that section says. Run step 10's review over its diff before opening it, and label it `from-playground`.
- Finish the port against that PR: add `mjswan = { git = "https://github.com/ttktjmt/mjswan", rev = "<head sha>" }` under `[tool.uv.sources]`, `uv lock --upgrade-package mjswan`, then sync, build and parity as usual. The web client builds through nodeenv on first use.
- The task's pull request is a draft labelled `needs-mjswan` that links the mjswan PR. Nothing is published or merged while the pin is there: the `released-mjswan` check refuses a git-sourced mjswan on `main`. When the mjswan PR gets new commits, move the pin to its new head and verify again.
- When only some simulations need the change, the rest land without it. The waiting ones go in a draft of their own, from a branch cut from the task's (`<task branch>-mjswan`), labelled `needs-mjswan` and linking the mjswan PR and the task's. Its diff narrows to them once the task's pull request is merged.
- Finish it once `origin/main` locks an mjswan release that contains the change: merge `origin/main`, drop the pin, `uv lock`, then steps 6 to 8 and 10 again, refilming the preview against the release; remove `needs-mjswan` and mark the PR ready. If the change touched `src/mjswan/template/`, say in the PR that publishing has to wait until mjswan Cloud serves that engine.

## 10. Final review (replaces "Report")

Only once steps 6 to 8 pass, with a step 9 pin or without. Stage everything (`git add -A`) and read `git diff --cached origin/main` as its reviewer would:

- the README follows reference/task-readme.md, and "What differs from upstream" names every term the port dropped, replaced or skipped;
- every source is pinned to a commit or revision, and nothing binary but `assets/<id>.gif`, and nothing from `.cache/` or `dist/`, is staged;
- English throughout, no em dashes, comments only for a non-obvious why.

Then run mjswan's `simplify-comments` on the same staged diff and apply what it finds: follow `$MJSWAN/.claude/commands/simplify-comments.md` with that diff as the target. It names `ponytail:ponytail-review`; invoke it when it is in your skill list, otherwise read `.cache/ponytail/skills/ponytail-review/SKILL.md` from a pinned clone:

```sh
[ -d .cache/ponytail ] || git clone -q --filter=blob:none https://github.com/DietrichGebert/ponytail .cache/ponytail
git -C .cache/ponytail -c advice.detachedHead=false checkout -q 5c21d7e9c6cf7ac5bb62dbabcd325223125314ad
```

It touches comments and docstrings only. Finish with `make format` and `make test`.

## 11. Pull request

- Commit on the task's branch as `Add <id>: <what it shows>` (`Add musclemimic: a 354-muscle body tracking a clip with the public checkpoint`), push, and open a pull request against `main` under the same title.
- The body, in this order: what the task shows, its sources and licenses; its simulations as a table, each with its scene and policy, or left out with the step, the error verbatim and what would unblock it; the `mjswan info` tree; terms traced and the parity result for each scene (`report.summary()` verbatim if anything failed), with the terminations parity left unchecked; terms skipped or dropped, each with what unblocks it; what differs from upstream; the preview's rounds, each failure with its fix and then the clean round's `checks passed` line for each cut; and what its frames show, adding that browser behaviour past those seconds is unverified: parity matches the graphs to mjlab, while mjlab integrates with `mujoco_warp` and the browser runs MuJoCo's WASM build.
- With a step 9 pin: a draft, labelled `needs-mjswan`, linking the mjswan PR.
- Wait for the pull request's checks to finish on the pushed head, and fix a red one before reporting.
- Never merge it and never publish. Tell the user the PR URL, anything you stopped on, and `uv run mjswan publish dist/<id>` for the author to run.
