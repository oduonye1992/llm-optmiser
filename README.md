# Bernard prompt optimizer

This tool helps you test new prompts for Bernard, a voice in a two-person dating call. Jev picks **speak** or **silent**. If Jev picks **speak**, Luna writes one short line.

You review call moments in Braintrust. This tool copies only complete reviews to a private dataset. It checks a model judge against your old reviews. Then DSPy GEPA tries new Jev and Luna instructions. It saves two candidate prompt files. It never edits the live agent.

The code, seed prompts, and examples in this repo are public. The examples are made up. Keep call text, keys, review reasons, reports, and candidate prompts out of Git.

The [Philomena Cunk interview research](docs/philomena-cunk-research.md) links to aired transcripts and maps the interview style to Jev and Luna. It includes original examples, not show scripts.

The Button examples draw on [Quickest Rejection in Button History?](https://www.youtube.com/watch?v=DUPXwNlp1mA), [Reject Your Date With One Press of a Button](https://www.youtube.com/watch?v=EoJjvEwWuxE), and [Men & Women Compete in Brutal Speed Dating Game](https://www.youtube.com/watch?v=J7Ku0LiaV1w). The Button asks about a fact, a choice, or a weak answer that just came up. In [I Voiced the Button](https://www.youtube.com/watch?v=HgeZGYy5LEQ), Cody Ko says he watched the date and picked when to cut in. The new cases use these moves, not lines from the show.

The new cases are made up and marked `source: synthetic`. Their `example_line` shows a possible line for review. The fixed scores check the choice and hard rules; they do not judge humor. Only your real call reviews go into `Bernard reviewed moments`.

## Quick start

Install [uv](https://docs.astral.sh/uv/). Then run:

```sh
uv sync
cp .env.example .env
uv run pytest
```

Fill in the three keys in `.env`. The local file is ignored by Git. `config.toml` holds the default model names, prompt paths, and GEPA call cap. You can change these in `config.toml` or use the matching `.env` fields:

| Job | Config key | `.env` name |
| --- | --- | --- |
| Jev choice model | `models.jev` | `JEV_MODEL` |
| Luna line model | `models.luna` | `LUNA_MODEL` |
| Luna effort | `models.luna_reasoning_effort` | `LUNA_REASONING_EFFORT` |
| Luna reply token cap | `models.luna_max_completion_tokens` | `LUNA_MAX_COMPLETION_TOKENS` |
| Line judge model | `models.judge` | `JUDGE_MODEL` |
| GEPA reflection model | `models.reflection` | `REFLECTION_MODEL` |
| Jev seed prompt | `prompts.jev` | `JEV_PROMPT` |
| Luna seed prompt | `prompts.luna` | `LUNA_PROMPT` |
| GEPA call cap | `gepa.max_metric_calls` | `MAX_METRIC_CALLS` |

The seed prompts began as copies of `livekit-agent/src/bernard_prompts.py` in Pineapple. The source file SHA-256 was `597ed937d45d5ca4a041bc87de7b95288ed72accc18ad4131cb9b49debd4c70f` when they were copied. The offline seeds now give Jev only the choice job and Luna only the line job. They differ from the live prompts. The runner uses the same request fields as Bernard's `decision_client.py`. A test checks the request fields and call context when that local agent checkout is present.

## What is one example?

One **call moment** is one dataset row. It holds the call context at one Jev choice, the old choice, any old Luna line, and your review. One call can give several reviewed moments. Each call is kept in one train, check, or final test group. A Braintrust **experiment** is made when you compare a prompt pair on the held-out test rows. It is not made for every call.

Raw Bernard calls stay in Braintrust Logs. The `Bernard reviewed moments` dataset is a copy of only the moments you fully review. The tool uses the original log row ID as the dataset row ID. Running `sync` again updates that row instead of adding a second copy.

`examples/fake_reviews.jsonl` shows the private snapshot shape with 24 made-up moments. `examples/bernard_safety_cases.jsonl` holds made-up fixed safety checks. Neither file is a real call export.

## How to review a call moment

In your Braintrust project, open **Settings → Human review**. Add these two **categorical** scores:

| Score | Pass | Fail | When to use it |
| --- | --- | --- | --- |
| `Jev Choice` | `1` | `0` | Every choice you want in the dataset |
| `Luna Line` | `1` | `0` | Only when Jev spoke |

If these scores already exist, check their values. The script accepts only exact `1` and `0`. The Pineapple `pineapple-bernard` project had both scores set to those values on 2026-10-15. It also had an empty `Bernard reviewed moments` dataset.

Open a **decision** row in Logs. Do not review the whole-call transcript row for this tool. Choose Pass or Fail for `Jev Choice`. A Jev Fail means the other choice was better. If Luna spoke, choose Pass or Fail for `Luna Line` too.

Add one row comment with plain reasons. Use these exact labels:

```text
Jev: He cut in while Leo was still answering Ava's question.
Luna: The line felt like a stock joke. Ask about their cooking banter.
```

A silent row only needs the `Jev:` line. A spoken row needs both lines. The reason can say what worked on a Pass and what should change on a Fail. Blank reasons, missing scores, bad score values, and rows with an unknown request shape do not enter the dataset. A bad value or broken reviewed row stops the sync with an error.

Braintrust's [human review guide](https://www.braintrust.dev/docs/annotate/human-review) explains the review screen. Its [comments guide](https://www.braintrust.dev/docs/kb/fetching-human-review-comments-via-api-and-btql) shows how row comments come back through the API.

## Run the steps

Real call text can reach the judge and GEPA reflection model. First make sure those providers may receive it. For Pineapple, close the private-data gate for this use. Then set `PRIVATE_DATA_APPROVED=true` in your ignored `.env`. The tool blocks all real-data commands while this value is false.

```sh
uv run llm-optmiser sync
uv run llm-optmiser check-judge
uv run llm-optmiser optimize
uv run llm-optmiser evaluate --run runs/<run-id>
```

Use `--config path/to/config.toml` **before** the command to load another config. Use `--snapshot path/to/reviews.jsonl` **after** a command to use another private snapshot.

- `sync` reads paged project logs. It writes complete reviews into the private Braintrust dataset. It also writes `data/reviewed.jsonl` and a small version file. That folder is ignored by Git. The command prints counts, the Braintrust version, and a SHA-256 hash of the fixed snapshot. It does not print call text.
- `check-judge` tests the judge on old Luna lines in the check split. It hides the old Pass or Fail value from the judge. It stops if agreement is below 80% or either review class is missing.
- `optimize` needs at least 20 complete reviews from 6 calls. It also needs 6 wanted speaks, 6 wanted silences, 2 Luna Pass reviews, and 2 Luna Fail reviews. It keeps whole calls together when it splits the rows. GEPA uses the train and check rows. It gets a number and a written reason for each try. It does not see the final test rows. The run saves `candidate_jev.txt`, `candidate_luna.txt`, and `manifest.json` under ignored `runs/<run-id>/`.
- GEPA rejects a draft that asks Jev to write a line, asks Luna to choose silence, or copies a known name or long phrase from reviewed calls. The chosen prompts face the same check. A text check can miss a reworded private story, so read chosen prompts before sharing them.
- `evaluate` checks that the snapshot and seed prompts still match that run. It tests the old pair and the candidate pair on the **same** held-out rows. It also runs fixed Bernard safety cases. It writes two Braintrust experiments and `report.json`. A provider error or missing score fails the command.

If GEPA keeps the seed prompt, `evaluate` says `prompt_changed=false`. It skips the held-out calls and experiments because there is no new prompt to compare. The run record also saves Luna's reasoning effort and reply token cap. `evaluate` stops if either setting changed.

The score gives Jev a full point for a correct silence. A wrong choice gets zero. A correct speak gets half a point plus up to half a point from the line judge. A hard-rule miss gets zero. A candidate counts as better only if it beats the old pair on the held-out rows and has no hard-rule miss. The judge is a model that reads your written reasons; GEPA does not come with your taste built in. See the [DSPy GEPA guide](https://dspy.ai/3.4.1/getting-started/gepa-optimization/) for the search method.

## Applying a candidate

Read both candidate files, the report, and sample outputs. This offline check cannot prove how Bernard sounds or when he cuts in on a live call. To use a candidate, copy it into the live agent in a separate change. Then run the agent's saved cases and a real two-person call check. This tool does not deploy Bernard.

## Checks and data safety

```sh
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

`.gitignore` blocks `.env`, `data/`, `runs/`, caches, and virtual environments. The public repo contains no real call text. Check `git ls-files` before each public push. Do not paste real reviews into issues or pull requests.

License: MIT.
