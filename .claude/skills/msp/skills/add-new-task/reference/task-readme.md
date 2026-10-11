# Task README

Every task's `README.md` has the same frame; the five existing ones are the examples. It says what a visitor sees, how to run it, where every input comes from, and what is not upstream's.

~~~markdown
# <Name> (`<id>`)

<img src="../../../assets/<id>.gif" width="480" alt="<Name> preview"/>

Source: <upstream repo URL> (<license>) ·
<policy and clips from <URL> (<license>), when they live elsewhere> ·
project page: <URL>

> <Authors>.
> [<Paper title>](<arXiv URL>).
> <Venue>, <year>. arXiv:<number>.

<One or two sentences: what the visitor sees and can do in the browser.>

## Run

```sh
uv sync --extra <id>            # an extra only: what it registers
uv run msp run <id>
```

<What the first build fetches and where it lands (`.cache/`), the `MJSWAN_<NAME>_ROOT`
variable of each checkout for one the reader already has, and any login it needs
(`hf auth login`).>

| From `<upstream>` | Used as |
|---|---|
| `<path>` | <the scene, the policy, the joint order and default pose, ...> |

## What differs from upstream

<Every term dropped, replaced or skipped and every substitution, each with its reason.
Training-only terms count.>

## License

<Only when the license is not permissive or assets carry terms of their own: those terms,
and that the demo is published only by its author.>
~~~

- The Source line credits everything the demo uses, each with its license: the code, the checkpoint and the clips when they live elsewhere, and a motion's origin.
- The preview GIF from step 7 opens the README, under the title. Drop the image when the preview did not film.
- With several simulations, list them after the opening sentence, one line or table row per scene saying what it does, and give "Run" a table of each scene's policies, upstream task and checkpoint (`jumper`, `microduckpg`). Name the simulations upstream ships that the task leaves out, and why.
- Drop the quote block when there is no paper, and the License section when everything is permissive.
- Add a section only when it earns its place, as the existing ones do: "What the policy reads" for the observation layout (all but `wbc`), "How it behaves" (`pacman`), "The one known gap" (`microduck`).
- Wrap prose at about 90 columns, separate the sources with `·`, and use no em dashes.
