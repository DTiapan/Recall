# Craft setup (Recall)

Recall adopts [Craft v0.1](https://github.com/DTiapan/craft) for workflow routing and the [engineering ledger](engineering-ledger/INDEX.md). Craft skills are **not** vendored in this repo.

Project pointer: [craft.project.yaml](../craft.project.yaml)

---

## New machine checklist

Run once after cloning Recall on a laptop or CI dev box:

```bash
# 1. Recall project
git clone git@github.com:DTiapan/Recall.git
cd Recall
uv venv && source .venv/bin/activate
uv pip install -e ".[formats,dev]"

# 2. Addy agent-skills (from Recall project root so skills land in .agents/skills/)
cd Recall
npx skills add addyosmani/agent-skills

# 3. Craft (router + ledger skills)
git clone git@github.com:DTiapan/craft.git ~/.cursor/skills/craft
ln -sf ~/.cursor/skills/craft/skills/* ~/.cursor/skills/

# 4. Verify Craft setup (run from Recall root OR any cwd)
bash ~/.cursor/skills/craft/skills/craft-adopt/scripts/verify-deps.sh
bash ~/.cursor/skills/craft/scripts/smoke-test.sh

# 5. Optional: local agent memory (Battery, separate repo)
# git clone git@github.com:DTiapan/battery.git && cd battery && uv sync
# uv run battery onboard   # run from Recall repo root
```

---

## What is already in this repo (committed)

| Artifact | Path |
|----------|------|
| Engineering ledger | `docs/engineering-ledger/` |
| ADRs | `docs/decisions/` |
| Agent rules | `AGENTS.md` (Craft section) |
| Project manifest | `craft.project.yaml` |

Do **not** copy Craft or Addy `SKILL.md` files into Recall git. Agents load them from `~/.cursor/skills/` and `~/.agents/skills/`.

---

## Session ritual

**Start:** Read [engineering-ledger/INDEX.md](engineering-ledger/INDEX.md) → route via `using-craft` → read one upstream skill.

**End:** Append DR/LL → update phases if a gate cleared → update INDEX.

Lifecycle map: [github.com/DTiapan/craft/blob/main/docs/lifecycle-map.md](https://github.com/DTiapan/craft/blob/main/docs/lifecycle-map.md)

---

## Troubleshooting verify-deps

**"Missing required Addy skills"** after setup:

1. Run `npx skills add addyosmani/agent-skills` from **Recall project root** (creates `.agents/skills/`).
2. Symlink Craft: `ln -sf ~/.cursor/skills/craft/skills/* ~/.cursor/skills/`
3. Pull latest Craft: `git -C ~/.cursor/skills/craft pull`
4. Re-run: `bash ~/.cursor/skills/craft/scripts/smoke-test.sh`

Craft v0.1.1+ only requires **core Addy pack skills** (19). Optional: `implement`, `diagnose`, `domain-modeling`, `codebase-design`.

**"craft.manifest.yaml not found"**: clone Craft to `~/.cursor/skills/craft`, not a partial copy.

---

```bash
npx skills update addyosmani/agent-skills
git -C ~/.cursor/skills/craft pull
bash ~/.cursor/skills/craft/skills/craft-adopt/scripts/verify-deps.sh
```

---

## Related projects

| Repo | Role |
|------|------|
| [Recall](https://github.com/DTiapan/Recall) | This project (RAG product) |
| [Craft](https://github.com/DTiapan/craft) | Router + ledger |
| [Battery](https://github.com/DTiapan/battery) | Optional MCP agent memory |
