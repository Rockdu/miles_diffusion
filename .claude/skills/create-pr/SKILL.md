---
name: create-pr
description: Create OR edit a pull request following the miles / miles-diffusion (radixark) conventions. Use whenever the user asks to open/create/raise a PR, write or update/edit a PR title or body/description on an existing PR, reword commits for a PR, or get a branch ready for review. Covers the hard English-only rule, conventional-commits, the PR body shape, and the PR checklist.
---

# Create a PR (miles / miles-diffusion conventions)

## Rule 0 — English only (hard rule)

**Every PR title, PR body, and commit message MUST be in English.** No exceptions,
even if the conversation with the user is in another language. Chat with the user in
their language; write the PR artifacts in English.

## Commit messages — conventional-commits

```
feat(rollout): add partial-rollout buffer
fix(megatron): correct fp32 marker on Qwen3.5 A_log
refactor(fsdp): tighten deterministic-mode comments
docs: clarify FP8 rationale for MoE
test: cover R3 routing replay
```

- Valid types only: `feat` `fix` `refactor` `perf` `docs` `test` `chore` `ci` `build` `style`.
  (A scope like `(fsdp)` is optional; `fsdp(...)` alone is **not** a valid type.)
- Subject line < 70 chars, imperative mood.
- Body explains **why**, not what — the diff already shows what.
- End the commit body with the Co-Authored-By trailer if the harness requires it.

Before opening the PR, check `git log <base>..HEAD` and reword any commit whose type
isn't valid. Offer to squash noisy WIP commits into a clean set.

## PR body shape

Keep it tight. Sections, in order (drop any that don't apply):

1. **What** — the change, in 1–3 bullets (the new flags/behaviour).
2. **Why** — the non-obvious rationale / the trap being solved.
3. **Validation** — if there were experiments: a settings table + a results table
   with real numbers, and any ablation that proves the key decision. Attach figures.
4. **Files** — one line per touched file with its role.
5. **Checklist** — paste the checklist below, ticking what's done.

## PR checklist (paste into the body)

```
- [ ] `pre-commit run --all-files` passes
- [ ] Added/updated tests for new behaviour
- [ ] `pytest -x` is green
- [ ] If launch flags changed, `python3 train.py --help` still parses
- [ ] If a public flag was added, it appears in the CLI reference docs
- [ ] If an example was added, it has a real walkthrough
```

Honestly report which items are **not** met (e.g. "no tests added", "pre-commit not
run locally") rather than ticking them blind.

## Opening the PR

1. Write the body to `PR_BODY.md` in the repo/workspace.
2. If `gh` is available and authed:
   `gh pr create --base <base> --head <branch> --title "<conventional title>" --body-file PR_BODY.md`
3. If no `gh` / token (common here — pushes go over SSH), give the user:
   - the compare URL: `https://github.com/<owner>/<repo>/compare/<base>...<branch>?expand=1`
   - the title + `PR_BODY.md` to paste, and a reminder to drag in any figure.
   Do **not** invent a token or push secrets.

## Editing an EXISTING PR (title / description)

Same English-only rule and body shape apply to edits.

1. Find the PR: `gh pr view <number|url|branch> --json number,title,body,url` (or ask
   the user for the PR URL/number). To read the current body without `gh`, WebFetch
   the PR URL.
2. Revise `PR_BODY.md` — usually augment (add a Validation section, fix the checklist),
   don't blindly overwrite; preserve anything the author already wrote.
3. Apply:
   - with `gh`: `gh pr edit <number|url> --title "<title>" --body-file PR_BODY.md`
     (omit `--title` to keep it; `--body-file` replaces the whole body).
   - no `gh` / token: hand the user the revised `PR_BODY.md` to paste into the PR's
     **Edit** box, plus the PR URL. You cannot edit it for them without a token.
4. If commits also need fixing, reword/squash on the branch and `git push --force-with-lease`
   — the open PR updates automatically.

## Reference

Full source of these conventions: miles core `docs/developer/contributing.md`
(radixark/miles). miles-diffusion has no separate PR template — it reuses these.
