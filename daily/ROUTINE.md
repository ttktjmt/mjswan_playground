# Daily run

What the "mjswan playground daily" routine does each morning. Its prompt only points here, so changing the routine is a pull request against this file.

Each run is a fresh cloud session with ttktjmt/mjswan_playground and ttktjmt/mjswan checked out. It asks nothing: what this file and the backlog leave open is a stop. The parts run in order, and the run ends only after part G has checked its work and the report is published.

## A. Scout and rank the backlog

The backlog is ranked on two things weighed equally: how much attention a task draws, and how likely an unattended run is to port it without a change to mjswan that only it would use. A task that is famous but hard, or easy but little known, waits behind one that is fairly well known and fairly easy.

1. Search for repositories built on mjlab: the `mjlab` topic on GitHub (github.com/topics/mjlab, sorted by stars and by recently updated), a GitHub search for `mjlab` sorted by stars, mjlab's "Show and tell" discussions, and the web for posts, project pages and papers about mjlab releases (X, Reddit, Hacker News, YouTube, arXiv). Read GitHub's web pages with WebFetch and public repositories with plain `git`: the session's GitHub tools reach only the attached repositories, and the proxy refuses `curl` to github.com and api.github.com.
2. Vet each repository that is not already in the backlog, in `mjswan_playground.registry.ALL_TASKS`, in a PR, or in a `daily-task-skipped` issue:
   - its env config is mjlab's `ManagerBasedRlEnvCfg`, registered with mjlab or buildable from upstream's code for the port to register;
   - a trained policy is public at a pinned source (git, a release asset, the Hub, a public W&B run), and so is a tracking task's clip unless the ONNX carries it;
   - mjswan can run its actuators: mjlab's own classes, or a subclass that mjswan runs as its base (an `IdealPdActuatorCfg` subclass runs as PD within its effort limit), with the entry naming what that drops; or, for a robot a task here already runs data-only, the actuators that task runs (`microduck`: BAM in training, the MJCF's `<position>` servos in the browser), with the entry naming the training lags the policy needs;
   - the license of everything the build fetches is known. Any license will do, as long as the entry states it: an NC or SA one is the author's to weigh when publishing.
3. Score each one that passes, and every untried entry again with its stars refreshed:
   - `popularity`: 3 for 1,000 stars or more, or a post that reached a front page or 100k views; 2 for 200 to 999 stars, or a post with thousands of views; 1 for 50 to 199 stars; 0 below 50.
   - `ease`: 3 when mjswan runs it as it is, with mjlab's own action, actuator and command classes and at most a few `terms.py` rewrites or version shims; 2 when it needs several task-specific rewrites on the playground side (command bindings, stateful terms recast as commands, its own registry), a simulation the browser may not keep up with (physics above 500 Hz or control above 50 Hz), or a generic mjswan extension (msp:add-new-task step 9) with little else to rewrite; 1 for a generic extension on top of several rewrites, or a policy whose task or license is unverified; 0 when it needs a change to mjswan that only it would use, such as an action term its own repository defines, or has no usable policy.
4. Order `daily/backlog.yaml` by `popularity` plus `ease`, highest first. An entry with `ease` 0, which an unattended run will stop on, goes after every other. Break a tie with a robot or task type the playground does not have yet, then with more stars. A new entry gets every field filled, its two scores with their evidence, and a `found` line giving the date.
5. Commit on `claude/daily-backlog`, cut from `main` or with `main` merged in, and open or update its PR, labelled `daily-backlog`, listing each new or moved entry with its scores. No new entry and no change in order: change nothing.

Until a person merges that PR, the backlog for part D is the one on its branch.

## B. Tasks waiting on mjswan

For each open pull request labelled `needs-mjswan`, which links one ttktjmt/mjswan pull request:

- The mjswan PR closed without merging: close the task's PR with a comment saying so, and open a `daily-task-skipped` issue for its id.
- The mjswan PR has new commits: move the pin to its new head, `uv lock --upgrade-package mjswan`, build, rerun parity and push. The PR stays a draft.
- `main` locks an mjswan release that contains the mjswan PR's merge commit: finish the task as step 9 of msp:add-new-task says.
- Anything else: leave it.

## C. mjswan release bump

PyPI has a stable mjswan newer than the one `main` locks, and no open PR is labelled `mjswan-bump`: on `claude/mjswan-bump-<version>`, raise the floor in `pyproject.toml` (and the `mjlab` pin if mjswan's `mjlab` extra moved), `uv lock`, build every task, run `make test`, check every task's run with `scripts/record_preview.py --all --out-dir` a scratch directory (msp:add-new-task step 7 has the cloud flags), and open the PR labelled `mjswan-bump` with each task's check result.

## D. One new task

Skipped while three or more `needs-mjswan` PRs are open.

1. Pick the first backlog entry whose id is not in `mjswan_playground.registry.ALL_TASKS`, has no PR from `claude/daily-<id>` in any state, and has no open `daily-task-skipped` issue. None left: open one issue titled "Daily backlog is empty", unless one is open.
2. Run msp:add-new-task in unattended mode on branch `claude/daily-<id>`, with the entry as every answer: `repo` the target and `commit` the commit to pin, `id` the playground id, `task` the mjlab task, `license` the license, and `policy` and `notes` what they say. When the skill is not in your skill list, follow `.claude/skills/msp/skills/add-new-task/SKILL.md` directly. Its step 9 may open or reuse an mjswan PR, on branch `claude/playground-<id>` of ttktjmt/mjswan.
3. Label the PR `daily-task`.
4. On a stop: open, or update, an issue labelled `daily-task-skipped` and titled `<id>: <reason>`, with the step, the error verbatim and what would unblock it.

## E. Fix the routine

A step that failed, was skipped or needed a workaround because of the routine itself is fixed here, so the next run does not hit it. The routine is this file, msp:add-new-task under `.claude/skills/msp/`, and the scripts they run, such as `scripts/record_preview.py`; an instruction that cannot be followed in this environment is the routine's too. Not the routine's: a port's own bug, which msp:add-new-task fixes; an mjswan gap, which goes to its step 9; and a limit only a person can lift, such as a token or a network rule, which the report names with what would lift it.

1. Go over the run so far: every failure, retry, skipped step and workaround, and what caused it. Nothing traces back to the routine: change nothing.
2. Fix each one at its cause, with the smallest change to the instruction or script that would have prevented it. After a script change, `make test` passes.
3. Commit on `claude/daily-routine`, cut from `main` or with `main` merged in, and open or update its PR, labelled `daily-routine`. List each problem: the part and step, what happened with the error verbatim, and the change.

This run keeps following `main`'s routine: a fix takes effect once a person merges it. A task the problem stopped keeps its `daily-task-skipped` issue, which names that PR as what would unblock it.

## F. Review every change

Once parts A to E have made their last change, review all of them: the diff of every branch this run pushed, in either repository, against `origin/main` (`git diff origin/main...<branch>`).

1. Run `ponytail:ponytail-review` over each diff's code, every changed script and source file, not only its comments. Invoke it when it is in your skill list; otherwise read `.cache/ponytail/skills/ponytail-review/SKILL.md` from the pinned clone msp:add-new-task's step 10 makes. It only lists findings: apply each one that keeps behaviour as it is, and note why for any you leave.
2. Then run mjswan's `simplify-comments` over the same diffs, following `.claude/commands/simplify-comments.md` in the ttktjmt/mjswan checkout. It touches comments and docstrings only.
3. Commit what changed on each branch and push. Rerun what the changes touch: `make test`; a task's build, parity (`scripts/parity.py <id>`) and preview (msp:add-new-task steps 6 to 8) when its code changed, its own or a shared module it imports; and the mjswan PR's own tests when it did. Bring each PR's description up to date with any result that moved.

Keep each branch's findings, applied or left, and its `net:` line for the report.

## G. Check the run's work

Before the report, check every result of this run against GitHub and the files, never against memory. Fix what is off and check it again: a result is done only once its check passes. A fix made here goes through part F's review before it counts, and a fix to the routine itself goes on part E's branch. One that cannot be fixed this run leads the report as a failure, with what is wrong.

- Every branch this run pushed: its remote head is the local commit, and the working tree is clean.
- Every pull request this run opened or updated: it targets `main` and carries its labels (`daily-backlog`, `daily-routine`, `daily-task`, `mjswan-bump`, or `needs-mjswan` on a draft). Its checks have finished green on its current head: wait for them, up to 30 minutes. Root-cause a red check, fix it, push and wait again; one that is red on `main` too is noted, not fixed here. A `needs-mjswan` draft's `released-mjswan` check is red by design.
- The pull request of a task this run added or finished holds the whole task:
  - `src/mjswan_playground/<id>/`, with a README that opens with the GIF;
  - the registry line, the README row, `assets/<id>.gif` and its `PREVIEWS` entry;
  - nothing from `.cache/` or `dist/`.

  Its last preview round passed (`"passed": true` in `dist/preview/<id>.json`), and the committed GIF is that round's. A fresh worktree of the pushed head passes `make sync`, `make test` and `uv run msp build <id>`, so nothing the build needs was left uncommitted.
- `daily/backlog.yaml` on the backlog branch parses, and every new entry has every field.
- A stop's `daily-task-skipped` issue exists, with the step, the error verbatim and what would unblock it.
- Every branch part F reviewed carries the findings it applied, and the reruns they set off passed.

## Never

Merge anything; push to `main` of either repository; edit `daily/backlog.yaml` anywhere but on `claude/daily-backlog`; run mjswan's release workflow; run `mjswan login` or `mjswan publish`. Publishing stays with a person.

## Report

End every run, a stopped one included, by publishing one Artifact titled `Daily run <YYYY-MM-DD>`, written in Japanese for the owner, then close with its link and a two-line summary. Build it to be read at a glance, pictures first:

- At the top, one card per part, A to F: what it did, or why it did nothing. A stop leads, with the step, the error verbatim and what would unblock it.
- The new task, whether it reached a pull request or stopped:
  - msp:add-new-task's steps 00 to 11 as a strip, each marked done, skipped, or where the run stopped;
  - the preview GIF and its contact sheet, `dist/preview/<id>.png`, published as the page's own files;
  - a chart of the checked run from `dist/preview/<id>.json`: root height and tilt over the control steps, the filmed part shaded, every termination marked;
  - the preview's rounds as a table, each failure beside its fix;
  - parity, with the terms traced, dropped and skipped; its sources and licenses; and its pull request or issue, with the mjswan PR if one was opened or reused.
- The scout's new and moved backlog entries as a table: repository, `popularity` and `ease` with their evidence, license.
- The waiting PRs touched, each with what changed. After a bump, every task's check as a pass-or-fail grid, naming each first failure.
- Part E's problems, each with its cause and the change, and its pull request. Part F's findings for each branch, applied or left, with its `net:` line.

Build and publish it the way the Artifact tool's own instructions say, with the images as supporting files and the charts drawn from the data, and put nothing secret on the page. Then read the published page back with the Artifact tool, and republish until every card, image and chart is on it. Without the Artifact tool, end with the same content in Markdown and send the contact sheet as a file.
