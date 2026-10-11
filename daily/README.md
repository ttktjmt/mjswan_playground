# Daily tasks

A routine adds one task a day and opens a pull request for it, following [`ROUTINE.md`](ROUTINE.md). A task is one repository built on mjlab, with every simulation in it that mjswan can run, each a scene. The routine first scouts for such repositories and ranks [`backlog.yaml`](backlog.yaml) by how much attention each draws and how easily an unattended run can port it, weighed equally. It ends each run by fixing what in the routine got in its way, reviewing every change it made, checking its own work and simplifying the comments of every pull request it touched, then publishing a visual report, an Artifact in your claude.ai account. Merging and publishing stay with people, and merging a task is what puts it on the [GitHub Pages site](https://ttktjmt.github.io/mjswan_playground/).

- `ROUTINE.md`: what each run does, and what it never does.
- `backlog.yaml`: the queue. The scout's additions arrive as one standing pull request from `claude/daily-backlog`. Merge it to keep them, or edit the branch first; the routine already reads from it while it is open. An entry leaves the queue once its task is merged.
- The routine's fixes to itself arrive as one standing pull request from `claude/daily-routine`. A run follows `main`'s routine, so merge it for the next run to take them.
- [`released-mjswan.yml`](../.github/workflows/released-mjswan.yml): keeps a task that waits on an unreleased mjswan off `main`.
- [`deploy.yml`](../.github/workflows/deploy.yml): builds the tasks into one app on each pull request and publishes it to GitHub Pages from `main`, leaving out those `NOT_ON_SITE` in [`_site.py`](../src/mjswan_playground/_site.py) names. Its `ready` check fails when a task does not build or load.
- Labels: `daily-task`, `daily-backlog`, `daily-routine`, `daily-task-skipped`, `needs-mjswan` and `mjswan-bump` here; `from-playground` on ttktjmt/mjswan.

## Publishing a task

A task's pull request ships its preview with `WIP` in place of the link. To publish it:

```sh
make sync
uv run msp build <id>
uv run mjswan publish dist/<id>    # signs you in first if needed
```

Then replace `WIP` in its README row with the link, as the other rows have it, in a pull request.

## One-time setup

1. A cloud environment named `mjswan-daily`:
   - Network: Custom, with the default list included, plus:

     ```
     huggingface.co
     *.huggingface.co
     *.hf.co
     api.wandb.ai
     ```

   - Environment variables:

     ```
     CLAUDE_CODE_PLUGIN_DIRS=/home/user/mjswan_playground/.claude/skills/msp
     BASH_DEFAULT_TIMEOUT_MS=600000
     MUJOCO_GL=disable
     ANTHROPIC_MODEL=opus
     CLAUDE_CODE_SUBAGENT_MODEL=opus
     CLAUDE_CODE_SUBAGENT_MODEL_FORCE=1
     CLAUDE_CODE_EFFORT_LEVEL=max
     ```

     The last four run the session and every subagent it starts on Opus at max effort.
     The routine's editor has no effort setting, and a session with several repositories
     reads nothing but plugin keys from their `.claude/settings.json`.

2. In this repository's settings, set the Pages source to GitHub Actions, and in the `main` ruleset require the `released-mjswan` and `ready` status checks (`ready` is listed once `deploy` has run). Commits then reach `main` only through a pull request, yours included. Leave ttktjmt/mjswan's ruleset as it is: its release workflow pushes the version bump to `main`.
3. The routine, at claude.ai/code/routines:
   - Repositories: ttktjmt/mjswan_playground and ttktjmt/mjswan
   - Environment: `mjswan-daily`
   - Model: Opus, as `ANTHROPIC_MODEL` names it
   - Schedule: every day at 04:47 JST
   - Connectors: none
   - Prompt:

     ```
     Do today's daily run for ttktjmt/mjswan_playground exactly as daily/ROUTINE.md on its main branch says. You may push to claude/ branches of ttktjmt/mjswan_playground and ttktjmt/mjswan, open pull requests and issues in both, and add the labels that file names. Never merge, and never push to main.
     ```

4. Run it once by hand and read the run before leaving the schedule on. A finished run only means the session ended: check its report and the pull requests and issues it opened.
