


You said:
Uploaded image
Uploaded image
I want to create code contracts I can use with cursor and Claude. 

I want these to enforcr my ways of working and preferences. 

Must pass local ci before push

Must be pythonic and idiomatic 


Dictionary and vector over if elif then chains 

Separation of concerns 

No trashy util.py. Keep things separated 


Tailwind no css


Examples in images 


I need this to be flexible so I can drop it in all projects. 



ChatGPT said:
You want a portable “code contract” that you can drop into any repo and that both Cursor and Claude will follow. You also want local gates that block pushes when checks fail. You want idiomatic Python. You want clean structure. You want Tailwind-only on web work.

Use this layout in every repo.

/
  CONTRACT.md
  CLAUDE.md
  .cursor/
    rules.md
  .github/
    workflows/
      ci.yml
  scripts/
    check.sh
  pyproject.toml            # Python repos
  package.json              # Web repos
Below are copy-paste templates.


.cursor/rules.md
This gives Cursor a hard spine. It also stays portable.

scripts/check.sh
One entry point. Humans and bots both use it.

Make it executable.

chmod +x scripts/check.sh
Local push gate
This blocks pushes when checks fail. It stays repo-local.

Create .githooks/pre-push.

Now you get “must pass local CI before push” as an enforced rule, not a wish.

Python pyproject.toml baseline
This gives you “pythonic and idiomatic” as a default, with Ruff as the main gate.

[project]
name = "your_project"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = []

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "SIM", "RUF"]
ignore = ["E501"]

[tool.ruff.format]
quote-style = "double"

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_ignores = true
no_implicit_optional = true
strict = true

[tool.pytest.ini_options]
testpaths = ["tests"]
If you want the “dict and vector over if/elif” rule to bite harder, keep SIM enabled and add targeted review notes in CONTRACT.md. Linters do not catch every design smell. The contract does.

Web contract addendum for Tailwind-only
Add this file for web repos that use Tailwind.

Create WEB_CONTRACT.md.

Web rules apply.

Styling uses Tailwind only.
No CSS modules.
No styled-components.
No ad hoc CSS files.

Use server components by default.
Use client components only when interactivity requires it.
Validate all user input.
Use typed schemas at boundaries.
Reference it from CONTRACT.md when the repo has a UI.

A drop-in “contract pack” you can reuse
Copy these files into every repo with no edits. Edit only the stack lines.

CONTRACT.md
CLAUDE.md
.cursor/rules.md
scripts/check.sh
.githooks/pre-push

That set gives you behavior control, plus enforcement at push time.

If you tell me your default Python toolchain, uv or poetry or pip-tools, I will tailor scripts/check.sh and pyproject.toml to match it.


You said:
So try except pass or try except continue patter a are killing me now. I need to stop those 

I also need a better way to deal with stubs. Cursor will make placeholders that are impossible to find until they break

I want rules that if placeholders are created they must be logged in a placeholder.md file so we can systematically address ethme. 

Sometimes cursor uses non ascii characters especially in math. I don't want that. No delta shapes or tau 

Logging not print 


Logging needs to be consistent and correct not "print converted "


Functions should be self describing 


Minimal comments 

Always use doc strings. I use Google docs ring style 

I use read the docs for documentation 

I always want documentation to be more than swagger lists of functions. I want docs to be reabaleb by domain users not just software devs 



ChatGPT said:
Add these sections to your contract. Put them in CONTRACT.md, CLAUDE.md, and .cursor/rules.md. Keep one source of truth in CONTRACT.md. Treat the other two as mirrors.

## Error handling rules

No `except: pass`.
No `except Exception: pass`.
No `except: continue`.
No `except Exception: continue`.

Every exception path must do one of these.

Raise a specific exception.
Return a typed error value.
Log with `logger.exception(...)` and then raise.
Handle the exact expected exception type with a clear fallback.

Catching broad exceptions requires a comment in the code that names the risk and the reason. Use this exact marker.

BROAD_EXCEPT_OK: <one sentence reason>

## Placeholder rules

No silent stubs.
No placeholder returns like `return None` or `return {}` unless correct behavior.
No `TODO` without tracking.

If you create any placeholder, you must do all of this.

Add an entry to `PLACEHOLDERS.md` in the repo root.
Add a unique placeholder ID in code as `PLACEHOLDER:<id>`.
Add a failing test or an explicit NotImplementedError that includes the same ID.

Preferred placeholder shape.

`raise NotImplementedError("PLACEHOLDER:P-001 <short reason>")`

`PLACEHOLDERS.md` entry must include ID, file path, symbol, owner, and exit condition.

## ASCII rules

Use ASCII only in source files.
No non-ASCII symbols in identifiers, comments, docstrings, logs, or docs.
No Greek letters. No Δ, τ, λ. Use `delta`, `tau`, `lambda` as plain ASCII words if needed.

Reject commits that introduce non-ASCII.

## Logging rules

No `print` in library or app code.
Use the standard logging module.
Use a module-level logger.

`logger = logging.getLogger(__name__)`

Logs must state an action and a result.
Logs must include key fields as structured extras when useful.
No vague messages like "converted" or "done".

Use these patterns.

`logger.info("Converted %s rows for %s", n_rows, dataset_name)`
`logger.warning("Missing %s for %s. Using default.", field, entity_id)`
`logger.exception("Failed to parse record %s", record_id)`

## Function and docstring rules

Functions must be self-describing.
Use names that encode intent and units. Example: `load_prices_usd_daily`.

Keep comments rare.
Use docstrings on every public function, class, and module.
Use Google-style docstrings.

Docstrings must include intent, inputs, outputs, and failure modes.
Docstrings must include units and timezones when relevant.
Docstrings must avoid filler.

## Documentation rules

Docs target domain users first.
Docs must explain workflows, concepts, and decisions.
Docs must include at least one end-to-end example per major feature.

No docs that read like Swagger output.
Reference API docs can exist, but narrative docs must lead.

Docs live under `docs/` and build with Read the Docs.

Each new feature must add or update one narrative page.

Minimum doc set for any repo with users.

docs/index.md
docs/overview.md
docs/quickstart.md
docs/concepts.md
docs/howto.md
docs/faq.md
docs/reference.md
Now add enforcement so the rules bite.

Add this to scripts/check.sh after the Python block starts.

python - <<'PY'
import sys
from pathlib import Path

bad = []
for p in Path(".").rglob("*"):
    if p.is_dir():
        continue
    if any(part in {".git", ".venv", "node_modules", "dist", "build", "__pycache__"} for part in p.parts):
        continue
    if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".parquet", ".feather"}:
        continue
    try:
        b = p.read_bytes()
    except Exception:
        continue
    try:
        b.decode("ascii")
    except UnicodeDecodeError:
        bad.append(str(p))
if bad:
    print("Non-ASCII detected in source files:")
    for f in bad:
        print(" -", f)
    sys.exit(1)
PY

python - <<'PY'
import re, sys
from pathlib import Path

deny = [
    r"\bexcept\s*:\s*(pass|continue)\b",
    r"\bexcept\s+Exception\s*:\s*(pass|continue)\b",
]
paths = []
for p in Path(".").rglob("*.py"):
    if any(part in {".git", ".venv", "node_modules"} for part in p.parts):
        continue
    txt = p.read_text(encoding="utf-8", errors="ignore")
    for pat in deny:
        if re.search(pat, txt):
            paths.append((str(p), pat))
if paths:
    print("Forbidden exception pattern detected:")
    for f, pat in paths:
        print(f" - {f} matched {pat}")
    sys.exit(1)
PY
Create PLACEHOLDERS.md template once.

# Placeholders

This file tracks every placeholder. No placeholder stays untracked.

## Format

ID: P-###
File:
Symbol:
Owner:
Created:
Reason:
Exit condition:
Link:

## Entries
Add one more check to scripts/check.sh to force tracking.

python - <<'PY'
import re, sys
from pathlib import Path

ids = []
for p in Path(".").rglob("*.py"):
    if any(part in {".git", ".venv", "node_modules"} for part in p.parts):
        continue
    txt = p.read_text(encoding="utf-8", errors="ignore")
    ids += re.findall(r"PLACEHOLDER:(P-\d{3})", txt)

tracked = set()
ph = Path("PLACEHOLDERS.md")
if ph.exists():
    tracked = set(re.findall(r"\b(P-\d{3})\b", ph.read_text(encoding="utf-8", errors="ignore")))

missing = sorted(set(ids) - tracked)
if missing:
    print("Untracked placeholders detected. Add these IDs to PLACEHOLDERS.md:")
    for x in missing:
        print(" -", x)
    sys.exit(1)
PY
Finally, add the logging and docstring expectations to Cursor’s rules so it stops defaulting to prints and comment clutter.

Put this block in .cursor/rules.md.

## Additional hard rules

No `print`. Use `logging`.
Every public symbol gets a Google-style docstring.
Minimize comments. Prefer good names and good docstrings.
No non-ASCII anywhere in the repo.
No `except: pass` or `except: continue`. No `except Exception: pass/continue`.
Any placeholder requires `PLACEHOLDER:P-###` plus an entry in `PLACEHOLDERS.md`.
If you want this to be even stricter, I can also give you a Ruff configuration that bans print and pushes you toward better exception handling. Ruff can catch a lot. Your custom checks cover the rest.


You said:
I want cursor to save all its outputs to me as files with all caps in the .cursor/ARCHIVE folder

Sometimes it junks up libraries with docs all over the place

ChatGPT said:
Cursor cannot force itself to auto-save every reply as a file unless the tool supports that behavior. You can still get close. You can make “write to file first” the default. You can make CI fail if the archive file does not exist. You can stop doc sprawl with hard path rules.

Add this contract section.

## Output archiving rule

Cursor must write every non-trivial output to a file under `.cursor/ARCHIVE/`.
Archive filenames use ALL CAPS.
Each archive file starts with a short header that names the task and the touched files.

“Non-trivial output” means any plan, design, contract change, multi-step instruction, or code guidance that spans more than a few lines.
If Cursor produces such output in chat, Cursor must also write the same content to an archive file.

Archive file naming format

`YYYYMMDD_HHMM_<TOPIC>_<SHORT_TAG>.md`

Example

`20260228_0839_ERROR_HANDLING_PLACEHOLDERS_RULES.md`
Add this to .cursor/rules.md.

## Archive first

For any non-trivial response, you must write it to `.cursor/ARCHIVE/` as an ALL CAPS filename.
After you write the file, you may paste a short summary in chat.
If you cannot write files, you must say so and still provide the full content in chat.

Archive files must start with

Title
Goal
Files touched
Checks run
Now enforce it with a local check. This check blocks pushes when Cursor “forgets” to archive.

Add this to scripts/check.sh.

python - <<'PY'
import re, sys
from pathlib import Path
from datetime import datetime, timedelta

archive_dir = Path(".cursor/ARCHIVE")
if not archive_dir.exists():
    print("Missing .cursor/ARCHIVE directory")
    sys.exit(1)

files = [p for p in archive_dir.glob("*.md")]
if not files:
    print("No archive files found in .cursor/ARCHIVE")
    sys.exit(1)

bad = [p.name for p in files if p.name != p.name.upper()]
if bad:
    print("Archive filenames must be ALL CAPS. Fix these:")
    for n in bad:
        print(" -", n)
    sys.exit(1)

pattern = re.compile(r"^\d{8}_\d{4}_.+\.MD$")
bad2 = [p.name for p in files if not pattern.match(p.name)]
if bad2:
    print("Archive filenames must match YYYYMMDD_HHMM_TOPIC_TAG.MD. Fix these:")
    for n in bad2:
        print(" -", n)
    sys.exit(1)
PY
That gives you a hard repo-level gate. Cursor can still chat. Cursor cannot ship unless it writes the archive file.

Now stop doc sprawl.

Put this in CONTRACT.md.

## Documentation placement rule

Docs live in `docs/` only.
No standalone markdown docs in `src/`, `app/`, or library modules unless the file is a README.md for that directory.

Allowed markdown outside docs

README.md at repo root
README.md inside a package folder
CHANGELOG.md at repo root
CONTRIBUTING.md at repo root
LICENSE at repo root
.cursor/ files

Everything else goes under `docs/`.
Enforce that with another check in scripts/check.sh.

python - <<'PY'
import sys
from pathlib import Path

allowed_roots = {
    "README.md",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "PLACEHOLDERS.md",
    "CONTRACT.md",
    "CLAUDE.md",
}

def allowed(path: Path) -> bool:
    if path.name in allowed_roots and len(path.parts) == 1:
        return True
    if "docs" in path.parts:
        return True
    if ".cursor" in path.parts:
        return True
    if path.name == "README.md":
        return True
    return False

bad = []
for p in Path(".").rglob("*.md"):
    if any(part in {".git", ".venv", "node_modules", "dist", "build", "__pycache__"} for part in p.parts):
        continue
    if not allowed(p):
        bad.append(str(p))

if bad:
    print("Markdown doc sprawl detected. Move these under docs/ or rename to README.md:")
    for f in bad:
        print(" -", f)
    sys.exit(1)
PY
If you want the archive rule to apply to specific tasks only, add a trigger phrase. Example. Any time you say “contract”, “rules”, “plan”, “design”, “architecture”, “runbook”, “checklist”, Cursor must archive. You can encode that in .cursor/rules.md as a hard requirement.


You said:
Ok I want you to give me one contract file again with all the additional stuff we discussed 

ChatGPT said:
Below is a single, consolidated CONTRACT.md you can drop into any repo.

It includes stack rules, Python rules, error handling bans, placeholder tracking, ASCII enforcement, logging standards, documentation standards, archive rules, and doc sprawl controls.

#

If you want, I can now generate the matching scripts/check.sh that enforces every clause above in one file.


You said:
Ok

ChatGPT said:
Cool. Here is the drop-in enforcement set that makes that contract real.

Create these files.

mkdir -p .cursor/ARCHIVE docs scripts .githooks
touch PLACEHOLDERS.md
PLACEHOLDERS.md


Entries
scripts/check.sh

#!/usr/bin/env bash
set -euo pipefail

deny_dirs=(".git" ".venv" "node_modules" "dist" "build" "__pycache__" ".mypy_cache" ".ruff_cache")

python - <<'PY'
import re, sys
from pathlib import Path

deny_dirs = {".git",".venv","node_modules","dist","build","__pycache__",".mypy_cache",".ruff_cache"}

def skip(p: Path) -> bool:
    return any(part in deny_dirs for part in p.parts)

def is_binary(p: Path) -> bool:
    return p.suffix.lower() in {".png",".jpg",".jpeg",".gif",".pdf",".zip",".parquet",".feather",".arrow",".avro",".mp4",".mov",".avi",".wav",".mp3",".ico",".woff",".woff2"}

bad = []
for p in Path(".").rglob("*"):
    if p.is_dir() or skip(p) or is_binary(p):
        continue
    try:
        b = p.read_bytes()
    except Exception:
        continue
    try:
        b.decode("ascii")
    except UnicodeDecodeError:
        bad.append(str(p))

if bad:
    print("Non-ASCII detected in source files:")
    for f in bad:
        print(" -", f)
    sys.exit(1)
PY

python - <<'PY'
import re, sys
from pathlib import Path

deny_dirs = {".git",".venv","node_modules","dist","build","__pycache__",".mypy_cache",".ruff_cache"}

def skip(p: Path) -> bool:
    return any(part in deny_dirs for part in p.parts)

deny = [
    r"\bexcept\s*:\s*pass\b",
    r"\bexcept\s*:\s*continue\b",
    r"\bexcept\s+Exception\s*:\s*pass\b",
    r"\bexcept\s+Exception\s*:\s*continue\b",
]

hits = []
for p in Path(".").rglob("*.py"):
    if skip(p):
        continue
    txt = p.read_text(encoding="utf-8", errors="ignore")
    for pat in deny:
        if re.search(pat, txt):
            hits.append((str(p), pat))

if hits:
    print("Forbidden exception pattern detected:")
    for f, pat in hits:
        print(f" - {f} matched {pat}")
    sys.exit(1)
PY

python - <<'PY'
import re, sys
from pathlib import Path

deny_dirs = {".git",".venv","node_modules","dist","build","__pycache__",".mypy_cache",".ruff_cache"}

def skip(p: Path) -> bool:
    return any(part in deny_dirs for part in p.parts)

ids = []
for p in Path(".").rglob("*.py"):
    if skip(p):
        continue
    txt = p.read_text(encoding="utf-8", errors="ignore")
    ids += re.findall(r"PLACEHOLDER:(P-\d{3})", txt)

ph = Path("PLACEHOLDERS.md")
tracked = set()
if ph.exists():
    tracked = set(re.findall(r"\b(P-\d{3})\b", ph.read_text(encoding="utf-8", errors="ignore")))

missing = sorted(set(ids) - tracked)
if missing:
    print("Untracked placeholders detected. Add these IDs to PLACEHOLDERS.md:")
    for x in missing:
        print(" -", x)
    sys.exit(1)
PY

python - <<'PY'
import re, sys
from pathlib import Path

deny_dirs = {".git",".venv","node_modules","dist","build","__pycache__",".mypy_cache",".ruff_cache"}

def skip(p: Path) -> bool:
    return any(part in deny_dirs for part in p.parts)

allowed_root = {
    "README.md",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "PLACEHOLDERS.md",
    "CONTRACT.md",
    "CLAUDE.md",
}

def allowed_md(p: Path) -> bool:
    if p.name in allowed_root and len(p.parts) == 1:
        return True
    if "docs" in p.parts:
        return True
    if ".cursor" in p.parts:
        return True
    if p.name == "README.md":
        return True
    return False

bad = []
for p in Path(".").rglob("*.md"):
    if skip(p):
        continue
    if not allowed_md(p):
        bad.append(str(p))

if bad:
    print("Markdown doc sprawl detected. Move these under docs/ or rename to README.md:")
    for f in bad:
        print(" -", f)
    sys.exit(1)
PY

python - <<'PY'
import re, sys
from pathlib import Path

archive_dir = Path(".cursor/ARCHIVE")
if not archive_dir.exists():
    print("Missing .cursor/ARCHIVE directory")
    sys.exit(1)

files = list(archive_dir.glob("*.md")) + list(archive_dir.glob("*.MD"))
if not files:
    print("No archive files found in .cursor/ARCHIVE")
    sys.exit(1)

bad = [p.name for p in files if p.name != p.name.upper()]
if bad:
    print("Archive filenames must be ALL CAPS. Fix these:")
    for n in bad:
        print(" -", n)
    sys.exit(1)

pat = re.compile(r"^\d{8}_\d{4}_.+\.MD$")
bad2 = [p.name for p in files if not pat.match(p.name)]
if bad2:
    print("Archive filenames must match YYYYMMDD_HHMM_TOPIC_TAG.MD. Fix these:")
    for n in bad2:
        print(" -", n)
    sys.exit(1)
PY

if [ -f pyproject.toml ]; then
  echo "Python checks"
  python -m ruff format .
  python -m ruff check .
  python -m mypy .
  python -m pytest -q
fi

if [ -f package.json ]; then
  echo "Web checks"
  npm run -s lint
  npm run -s typecheck
  npm run -s test --if-present
  npm run -s build --if-present
fi

echo "OK"
.githooks/pre-push

#!/usr/bin/env bash
set -euo pipefail
./scripts/check.sh
Enable the hook.

chmod +x scripts/check.sh .githooks/pre-push
git config core.hooksPath .githooks
From here, you get hard failures for non-ASCII, trash exception patterns, untracked placeholders, doc sprawl, and missing archive hygiene.


You said:
What about the pattern of keeping all the memory in markdown? Should I enforce that?

ChatGPT said:
Yes. Enforce it.

You already use Cursor as a workflow engine. Memory drift kills that. Markdown memory gives you one place to look. Git gives you history. CI gives you enforcement.

Make it narrow. Put only durable facts and decisions in memory. Keep everything else in code and docs.

Add this to CONTRACT.md.

# 13. Project Memory Rule

Project memory lives in markdown.
Project memory lives in `docs/memory/`.

No durable decisions live only in chat threads.
No durable decisions live only in code comments.
No durable decisions live only in tickets.

“Durable” means:
- Architecture decisions
- Standards
- Conventions
- Data contracts
- Interface contracts
- Runbooks
- Operational assumptions
- Known risks
- Open questions that block design

Minimum memory set:

docs/memory/README.md
docs/memory/DECISIONS.md
docs/memory/ASSUMPTIONS.md
docs/memory/RISKS.md
docs/memory/OPEN_QUESTIONS.md
docs/memory/CHANGELOG.md

Rule for any meaningful change:
Update at least one memory file.

Decision format:
Date
Decision
Context
Alternatives considered
Tradeoff
Impact
Owner
Review date
Then enforce it with checks.

First, allow docs/memory/ under your markdown placement rule. You already allow all docs/, so that part stays fine.

Second, enforce presence of the memory folder and files. Add this block to scripts/check.sh before lint.

python - <<'PY'
import sys
from pathlib import Path

required = [
    "docs/memory/README.md",
    "docs/memory/DECISIONS.md",
    "docs/memory/ASSUMPTIONS.md",
    "docs/memory/RISKS.md",
    "docs/memory/OPEN_QUESTIONS.md",
    "docs/memory/CHANGELOG.md",
]

missing = [p for p in required if not Path(p).exists()]
if missing:
    print("Missing required project memory files:")
    for p in missing:
        print(" -", p)
    sys.exit(1)
PY
Third, stop “memory” content from scattering into random places. You already block markdown outside docs. That handles most of it. Add one more rule. Ban “decision log” markdown outside docs/memory/.

python - <<'PY'
import re, sys
from pathlib import Path

deny_dirs = {".git",".venv","node_modules","dist","build","__pycache__",".mypy_cache",".ruff_cache"}

def skip(p: Path) -> bool:
    return any(part in deny_dirs for part in p.parts)

bad = []
for p in Path(".").rglob("*.md"):
    if skip(p):
        continue
    if "docs" in p.parts and "memory" in p.parts:
        continue
    txt = p.read_text(encoding="utf-8", errors="ignore")
    if re.search(r"^\s*#\s*(Decision|Decisions|ADR|Architecture Decision)", txt, re.IGNORECASE | re.MULTILINE):
        bad.append(str(p))

if bad:
    print("Decision-style markdown must live under docs/memory/. Move these files:")
    for f in bad:
        print(" -", f)
    sys.exit(1)
PY
Create the memory skeleton once.

docs/memory/README.md

# Project Memory

This folder holds durable project knowledge.

Update this folder when you change standards, architecture, interfaces, or operational assumptions.
docs/memory/DECISIONS.md

# Decisions

## Template

Date:
Decision:
Context:
Alternatives considered:
Tradeoff:
Impact:
Owner:
Review date:
docs/memory/ASSUMPTIONS.md

# Assumptions

List assumptions that affect design and operations.
Update when assumptions change.
docs/memory/RISKS.md

# Risks

List known risks and mitigations.
docs/memory/OPEN_QUESTIONS.md

# Open Questions

List questions that block design or delivery.
Each entry has an owner and next step.
docs/memory/CHANGELOG.md

# Memory Changelog

Track updates to project memory.
Date:
Change:
Owner:
This gives you a clean rule.

Chat stays ephemeral.
Markdown stays durable.
The repo stays searchable.
The workflow stays repeatable.


You said:
Ok. Everyone on my team has lots of projects. I want to distill those into makemsow that says here is what the project is. Here is the data source

So we don't have to schlep big code files and repos

I want to task cursor to describe the project in a way that could be reproduced 


The basic concepts seem easy but the data used and math seem harder

Including math is one of the most important things to me. 

Demonstrating that a website can show data on a graph is not compelling. It needs the ability to encode and leverage business logic  

ChatGPT said:
Yes. Make this a contract. Make this a file format. Make this a workflow.

You want a portable project description that carries three things. It names the data. It states the math. It encodes the business logic. A graph alone proves nothing.

Create one required artifact per project. Call it a Project Brief. Store it as markdown. Keep it small. Make it reproducible.

Use this directory in every project repo.

docs/projects/<PROJECT_ID>/

Require these files.

PROJECT.md
DATA.md
MATH.md
LOGIC.md
REPRO.md

Keep code out of it unless the math needs a reference implementation. When you include code, include only the smallest runnable snippet.

PROJECT.md template
# <Project name>

## Purpose
State the business problem. State the decision this supports. State the user role.

## Output
State what the project produces. Name the tables, metrics, or API outputs. State frequency and latency.

## Scope
State what the project covers. State what it excludes.

## Success criteria
State measurable success. Use domain metrics.

## Dependencies
State external systems. State owners.
DATA.md template
# Data

## Sources
Name each source. State where it lives. State access method. State refresh cadence.

## Tables and fields
State the canonical tables. State primary keys. State time fields. State units.

## Entity model
Define entities and relationships. Define grain. Define join keys.

## Quality rules
State null rules. State ranges. State dedupe logic. State late data handling.

## Data fingerprint
State how to re-fetch. State how to sample. State row counts for a reference pull.
MATH.md template
This file carries the “hard part.” Keep it ASCII only.

# Math

## Variables
Define symbols as ASCII names. Define units.

## Definitions
Define derived quantities. Define transforms.

## Equations
Write equations in plain text.

Example
score = w1 * z(price_spread) + w2 * z(volume_delta) - w3 * penalty

## Estimation
State how you fit parameters. State objective. State constraints.

## Uncertainty
State confidence intervals, calibration, or error bounds.

## Edge cases
State discontinuities, clipping, missing data policy, and guardrails.

## Validation
State invariants. State expected ranges. State sanity checks.
LOGIC.md template
This file encodes business logic. It must read like a domain handbook.

# Business logic

## Decision rule
State the decision. State the threshold. State tie-breakers.

## Rule engine
State rules in priority order. Use plain language plus a formal form.

Formal form example
If peak_mw_today > percentile(peak_mw_last_30_days, 0.95) then candidate_day = true.

## Exceptions
State exceptions. State override authority. State audit fields.

## Explainability
State which signals drive output. State which signals block output.
REPRO.md template
This file answers “can someone else recreate this.”

# Reproduction

## Inputs
State required credentials and environment variables.

## Pull steps
State exact commands or API calls to obtain data.

## Build steps
State commands to run checks. State commands to run the pipeline.

## Run steps
State how to run end to end. State run time assumptions.

## Expected outputs
State file names. State table names. State example row counts.

## Verification
State how to confirm correctness. State known test cases.
Now task Cursor with a repeatable instruction that forces this output.

Put this in .cursor/rules.md under a new section.

## Project distillation output

When I ask you to describe or distill a project, you must produce reproducible project docs.

You must create or update:
docs/projects/<PROJECT_ID>/PROJECT.md
docs/projects/<PROJECT_ID>/DATA.md
docs/projects/<PROJECT_ID>/MATH.md
docs/projects/<PROJECT_ID>/LOGIC.md
docs/projects/<PROJECT_ID>/REPRO.md

MATH.md must include variables, equations, estimation method, and validation rules.
LOGIC.md must encode business rules and thresholds.
All files must be ASCII only.
No placeholders. No TODO.

You must also write an archive file under .cursor/ARCHIVE with the same content summary.
Add a hard check so this never slips.

Add this block to scripts/check.sh.

python - <<'PY'
import sys
from pathlib import Path

root = Path("docs/projects")
if not root.exists():
    print("Missing docs/projects directory")
    sys.exit(1)

required = {"PROJECT.md","DATA.md","MATH.md","LOGIC.md","REPRO.md"}

bad = []
for d in root.glob("*"):
    if not d.is_dir():
        continue
    missing = sorted(required - {p.name for p in d.glob("*.md")})
    if missing:
        bad.append((str(d), missing))

if bad:
    print("Project docs incomplete. Each project folder needs PROJECT, DATA, MATH, LOGIC, REPRO.")
    for folder, missing in bad:
        print(folder + " missing " + ", ".join(missing))
    sys.exit(1)
PY
Use this prompt with Cursor when you want a project distilled.

Read the repo. Produce reproducible project docs.

Create docs/projects/PROJ-001/PROJECT.md, DATA.md, MATH.md, LOGIC.md, REPRO.md.

Do not copy large code. Do not paste notebooks.
Extract data sources, entity grain, math, and business rules.

Write ASCII only. No Greek symbols.
No placeholders. No TODO.
Use Google-style docstrings only in code files. Keep docs narrative for domain users.

After writing the docs, write .cursor/ARCHIVE/YYYYMMDD_HHMM_PROJ-001_PROJECT_DISTILLATION.MD.
Run scripts/check.sh.
This gives you what you asked for. Each project becomes a small, searchable, domain-readable spec. It also becomes reproducible. The math stays front and center. The business logic becomes explicit.


You said:
How should I handle data itself ? I have huge data files that I think are key to the demos I want to show. But I can't keep passing this data around 

Somehow I need an example schema and a way to pull the real data when needed

Maybe a data fingerprint?

The data comes from publish sources so I really need to preserve the url to get the raw data and then any transformations I do to the data and something like a recipe

ChatGPT said:
Treat data as a reproducible artifact, not a repo artifact.

You keep three things in git. You never keep the raw bulk data in git. You keep a schema, a source pointer, and a recipe. You keep a fingerprint so you can prove you used the same input.

Create a standard folder in every project.

data/
data/README.md
data/sources/
data/recipes/
data/schemas/
data/fingerprints/
data/samples/

You store only small samples. You store only text metadata. You store no big binaries.

Add this to your contract.

Raw data does not live in git.
Public source URLs must live in the repo.
Every dataset must have a fetch recipe.
Every dataset must have a schema.
Every dataset must have a fingerprint.

Now use a standard “data pack” per dataset. One dataset gets one ID. Example DATASET-001.

data/sources/DATASET-001.md
data/schemas/DATASET-001.schema.json
data/recipes/DATASET-001.py
data/fingerprints/DATASET-001.json
data/samples/DATASET-001.sample.csv

Source pointer file
data/sources/DATASET-001.md

# DATASET-001

Name: <dataset name>
Owner: Public source
License: <license or terms link>
Raw URL: <direct download URL>
Landing page: <human page URL>
Access method: HTTP GET
Auth: None

Update cadence: <daily/monthly/annual>
Expected format: <csv/parquet/json/zip>
Expected size: <approx>
Schema file
Use JSON Schema for tables. Keep it simple.

data/schemas/DATASET-001.schema.json

{
  "title": "DATASET-001",
  "type": "object",
  "properties": {
    "date": { "type": "string", "description": "YYYY-MM-DD" },
    "plant_id": { "type": "string" },
    "co2_tons": { "type": "number" }
  },
  "required": ["date", "plant_id", "co2_tons"]
}
Recipe
A recipe pulls raw data and produces a canonical, small set of “gold” tables. The recipe also writes a fingerprint file. Keep the recipe deterministic.

data/recipes/DATASET-001.py

This is the shape, not your exact code.

It must do this. Fetch. Verify. Normalize. Write outputs. Write fingerprint.

You should store outputs outside git. Use data/out/ and gitignore it.

Fingerprint
A fingerprint proves the exact input and the exact transform.

data/fingerprints/DATASET-001.json

{
  "dataset_id": "DATASET-001",
  "raw": {
    "raw_url": "<direct url>",
    "retrieved_at_utc": "2026-02-28T14:01:00Z",
    "http_etag": "<etag or null>",
    "http_last_modified": "<value or null>",
    "bytes": 123456789,
    "sha256": "<sha256 of raw bytes>"
  },
  "transform": {
    "recipe_path": "data/recipes/DATASET-001.py",
    "git_commit": "<commit hash>",
    "parameters": {
      "date_min": "2010-01-01",
      "date_max": "2025-12-31"
    },
    "outputs": [
      {
        "path": "data/out/DATASET-001_gold.parquet",
        "rows": 1234567,
        "columns": 42,
        "sha256": "<sha256 of output file>"
      }
    ]
  }
}
If the host supports ETag and Last-Modified, you store them. That helps reproducibility without storing the raw file.

Sample
Store a tiny sample that matches the schema. Keep it small. This powers docs and tests.

data/samples/DATASET-001.sample.csv

This file stays under 200 KB.

Data README
data/README.md

Explain the workflow in plain language for domain users. Explain where raw data lives. Explain how to fetch. Explain what the gold table means.

Git hygiene
Add this to .gitignore.

data/out/
data/raw/
*.parquet
*.zip

Keep exceptions only for tiny samples.

Doc integration
In your project docs, reference dataset IDs, not file paths. Your docs/projects/.../DATA.md lists dataset IDs, then links to data/sources/ and data/recipes/.

Enforcement
Add a check. Every dataset in docs/projects/*/DATA.md must have a matching source file, schema file, recipe file, and fingerprint file. That makes “data” a first class dependency.

This gives you what you asked for.

You preserve the URL.
You preserve the transform recipe.
You preserve the schema.
You preserve a fingerprint.
You avoid schlepping giant files.

If you want, I can also give you a single data/recipes/fetch.py runner that takes a dataset ID and runs the right recipe, so the whole team uses one command.



Copier also scaffolds. Copier can also update existing repos from the template later. That matters once your contract, checks, and docs evolve.

If you already have Cookiecutter in muscle memory, keep it. If you want long-term control across many repos, pick Copier.

Use a template repo either way. Call it something like project-template. Put your defaults in it. Put your contract in it. Put your checks in it. Put your memory skeleton in it. Put your data pack skeleton in it. Put your Cursor rules in it.

Include this base structure in the template.

CONTRACT.md
CLAUDE.md
.cursor/rules.md
.cursor/ARCHIVE/
scripts/check.sh
.githooks/pre-push
docs/
docs/memory/
docs/projects/PROJ-001/
data/
data/sources/
data/schemas/
data/recipes/
data/fingerprints/
data/samples/
PLACEHOLDERS.md
.gitignore

Then define one command as the team habit. Create repo. Enable hooks. Run checks. Add first project brief. Add first dataset pack.

If you want, I can draft a ready-to-use Copier template layout with a minimal copier.yml, plus the exact files with placeholders for project name, package name, and repo slug.


You said:
Yes please 

ChatGPT said:
Use Copier. Put one template repo in Git. Use it for every new project. Use it again later to roll out contract updates across old repos.

Create a repo named project-template. Put this in it.

project-template/
  copier.yml
  README.md
  template/
    {{ repo_slug }}/
      CONTRACT.md
      CLAUDE.md
      PLACEHOLDERS.md
      .gitignore
      .cursor/
        rules.md
        ARCHIVE/
          .keep
      .githooks/
        pre-push
      scripts/
        check.sh
      docs/
        index.md
        overview.md
        quickstart.md
        concepts.md
        howto.md
        faq.md
        reference.md
        memory/
          README.md
          DECISIONS.md
          ASSUMPTIONS.md
          RISKS.md
          OPEN_QUESTIONS.md
          CHANGELOG.md
        projects/
          PROJ-001/
            PROJECT.md
            DATA.md
            MATH.md
            LOGIC.md
            REPRO.md
      data/
        README.md
        sources/
          DATASET-001.md
        schemas/
          DATASET-001.schema.json
        recipes/
          DATASET-001.py
        fingerprints/
          DATASET-001.json
        samples/
          DATASET-001.sample.csv
      pyproject.toml.j2
      src/
        {{ package_name }}/
          __init__.py
          logging.py
      tests/
        test_smoke.py
Add this copier.yml at the repo root.

_min_copier_version: "9.3.0"

subdirectory: template

tasks:
  - "chmod +x scripts/check.sh .githooks/pre-push || true"

questions:
  repo_slug:
    type: str
    help: Repo folder name
  project_name:
    type: str
    help: Human name
  package_name:
    type: str
    help: Python package name
    default: "{{ repo_slug.replace('-', '_') }}"
  author_name:
    type: str
    help: Author name
    default: ""
  python_version:
    type: str
    help: Python version
    default: "3.11"
  use_python:
    type: bool
    help: Include Python scaffold
    default: true
  use_web:
    type: bool
    help: Include web rules only
    default: false
Put this README.md at the template repo root.

Use Copier to create a new repo from this template.

Install
pipx install copier

Create a project
copier copy . /path/to/new-repo

Update an existing repo later
copier update
Now add the template files under template/{{ repo_slug }}/.

CONTRACT.md

CLAUDE.md

You are a code assistant in this repo. Obey CONTRACT.md.

Start
Read CONTRACT.md.
State files you will touch.
State checks you will run.

During work
Keep diffs small.
Keep modules focused.
Avoid util.py.
Do not add dependencies unless approved.
Do not change schemas unless you provide a migration plan.

Finish
Run scripts/check.sh.
Fix every failure.
Do not claim success until checks pass.
Write an archive file in .cursor/ARCHIVE for non trivial output.
.cursor/rules.md

Follow CONTRACT.md. No exceptions.

Archive first rule
For any non trivial response, write it to .cursor/ARCHIVE as an ALL CAPS filename.
Then post a short summary in chat.
If file write fails, say so and still provide full content.

Hard rules
No print. Use logging.
No non ASCII.
No forbidden except patterns.
No silent placeholders.
Any placeholder must use PLACEHOLDER:P-### and must be tracked in PLACEHOLDERS.md.
Docs stay under docs. Only README.md may live near code.

Project distill rule
When asked to distill a project, create docs under docs/projects/<PROJECT_ID> with PROJECT, DATA, MATH, LOGIC, REPRO.
MATH includes variables, equations, estimation, validation.
LOGIC encodes business rules and thresholds.
ASCII only.
PLACEHOLDERS.md

# Placeholders

This file tracks every placeholder.

Format

ID: P-###
File:
Symbol:
Owner:
Created:
Reason:
Exit condition:
Link:

Entries
.githooks/pre-push

#!/usr/bin/env bash
set -euo pipefail
./scripts/check.sh
scripts/check.sh

#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import re, sys
from pathlib import Path

deny_dirs = {".git",".venv","node_modules","dist","build","__pycache__",".mypy_cache",".ruff_cache"}

def skip(p: Path) -> bool:
    return any(part in deny_dirs for part in p.parts)

def is_binary(p: Path) -> bool:
    return p.suffix.lower() in {".png",".jpg",".jpeg",".gif",".pdf",".zip",".parquet",".feather",".arrow",".avro",".mp4",".mov",".avi",".wav",".mp3",".ico",".woff",".woff2"}

bad = []
for p in Path(".").rglob("*"):
    if p.is_dir() or skip(p) or is_binary(p):
        continue
    try:
        b = p.read_bytes()
    except Exception:
        continue
    try:
        b.decode("ascii")
    except UnicodeDecodeError:
        bad.append(str(p))

if bad:
    print("Non-ASCII detected in source files")
    for f in bad:
        print(" -", f)
    sys.exit(1)
PY

python - <<'PY'
import re, sys
from pathlib import Path

deny_dirs = {".git",".venv","node_modules","dist","build","__pycache__",".mypy_cache",".ruff_cache"}

def skip(p: Path) -> bool:
    return any(part in deny_dirs for part in p.parts)

deny = [
    r"\bexcept\s*:\s*pass\b",
    r"\bexcept\s*:\s*continue\b",
    r"\bexcept\s+Exception\s*:\s*pass\b",
    r"\bexcept\s+Exception\s*:\s*continue\b",
]

hits = []
for p in Path(".").rglob("*.py"):
    if skip(p):
        continue
    txt = p.read_text(encoding="utf-8", errors="ignore")
    for pat in deny:
        if re.search(pat, txt):
            hits.append((str(p), pat))

if hits:
    print("Forbidden exception pattern detected")
    for f, pat in hits:
        print(f" - {f} matched {pat}")
    sys.exit(1)
PY

python - <<'PY'
import re, sys
from pathlib import Path

deny_dirs = {".git",".venv","node_modules","dist","build","__pycache__",".mypy_cache",".ruff_cache"}

def skip(p: Path) -> bool:
    return any(part in deny_dirs for part in p.parts)

ids = []
for p in Path(".").rglob("*.py"):
    if skip(p):
        continue
    txt = p.read_text(encoding="utf-8", errors="ignore")
    ids += re.findall(r"PLACEHOLDER:(P-\d{3})", txt)

ph = Path("PLACEHOLDERS.md")
tracked = set()
if ph.exists():
    tracked = set(re.findall(r"\b(P-\d{3})\b", ph.read_text(encoding="utf-8", errors="ignore")))

missing = sorted(set(ids) - tracked)
if missing:
    print("Untracked placeholders detected")
    for x in missing:
        print(" -", x)
    sys.exit(1)
PY

python - <<'PY'
import re, sys
from pathlib import Path

archive_dir = Path(".cursor/ARCHIVE")
if not archive_dir.exists():
    print("Missing .cursor/ARCHIVE directory")
    sys.exit(1)

files = list(archive_dir.glob("*.md")) + list(archive_dir.glob("*.MD"))
if not files:
    print("No archive files found in .cursor/ARCHIVE")
    sys.exit(1)

bad = [p.name for p in files if p.name != p.name.upper()]
if bad:
    print("Archive filenames must be ALL CAPS")
    for n in bad:
        print(" -", n)
    sys.exit(1)

pat = re.compile(r"^\d{8}_\d{4}_.+\.MD$")
bad2 = [p.name for p in files if not pat.match(p.name)]
if bad2:
    print("Archive filenames must match YYYYMMDD_HHMM_TOPIC_TAG.MD")
    for n in bad2:
        print(" -", n)
    sys.exit(1)
PY

python - <<'PY'
import sys
from pathlib import Path

required = [
    "docs/memory/README.md",
    "docs/memory/DECISIONS.md",
    "docs/memory/ASSUMPTIONS.md",
    "docs/memory/RISKS.md",
    "docs/memory/OPEN_QUESTIONS.md",
    "docs/memory/CHANGELOG.md",
]
missing = [p for p in required if not Path(p).exists()]
if missing:
    print("Missing required project memory files")
    for p in missing:
        print(" -", p)
    sys.exit(1)
PY

python - <<'PY'
import sys
from pathlib import Path

allowed_root = {
    "README.md",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "PLACEHOLDERS.md",
    "CONTRACT.md",
    "CLAUDE.md",
}

deny_dirs = {".git",".venv","node_modules","dist","build","__pycache__",".mypy_cache",".ruff_cache"}

def skip(p: Path) -> bool:
    return any(part in deny_dirs for part in p.parts)

def allowed_md(p: Path) -> bool:
    if p.name in allowed_root and len(p.parts) == 1:
        return True
    if "docs" in p.parts:
        return True
    if ".cursor" in p.parts:
        return True
    if p.name == "README.md":
        return True
    return False

bad = []
for p in Path(".").rglob("*.md"):
    if skip(p):
        continue
    if not allowed_md(p):
        bad.append(str(p))

if bad:
    print("Markdown doc sprawl detected")
    for f in bad:
        print(" -", f)
    sys.exit(1)
PY

if [ -f pyproject.toml ]; then
  echo "Python checks"
  python -m ruff format .
  python -m ruff check .
  python -m mypy .
  python -m pytest -q
fi

if [ -f package.json ]; then
  echo "Web checks"
  npm run -s lint
  npm run -s typecheck
  npm run -s test --if-present
  npm run -s build --if-present
fi

echo "OK"
pyproject.toml.j2

{% if use_python %}
[project]
name = "{{ project_name }}"
version = "0.1.0"
requires-python = ">={{ python_version }}"
dependencies = []

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "SIM", "RUF"]
ignore = ["E501"]

[tool.ruff.format]
quote-style = "double"

[tool.mypy]
python_version = "{{ python_version }}"
strict = true
warn_unused_ignores = true
no_implicit_optional = true

[tool.pytest.ini_options]
testpaths = ["tests"]
{% endif %}
src/{{ package_name }}/logging.py

import logging

def get_logger(name: str) -> logging.Logger:
    """Create a logger with the project standard.

    Args:
        name: Logger name.

    Returns:
        A configured logger.
    """
    return logging.getLogger(name)
tests/test_smoke.py

def test_smoke() -> None:
    assert True
Docs skeleton can stay plain. Keep it short at first. Put real content in it as you build.

Now create a new project from the template.

pipx install copier
copier copy /path/to/project-template /path/to/new-repo
cd /path/to/new-repo
chmod +x scripts/check.sh .githooks/pre-push
git config core.hooksPath .githooks
./scripts/check.sh
That gives you a repo with your gates, your memory, your project briefs, your data pack pattern, and your Cursor rules.




Here are the exact edits.

# copier.yml
_min_copier_version: "9.3.0"
subdirectory: template

tasks:
  - "chmod +x scripts/check.sh .githooks/pre-push || true"

questions:
  repo_slug:
    type: str
    help: Repo folder name
  project_name:
    type: str
    help: Human name
  package_name:
    type: str
    help: Python package name
    default: "{{ repo_slug.replace('-', '_') }}"
  author_name:
    type: str
    help: Author name
    default: ""
# CONTRACT.md add these sections

# Platform rule

This project supports Ubuntu Linux only.
This project supports Python 3.12 or newer only.
Do not add Mac or Windows guidance.
Do not add cross platform workarounds.

# Tone and claims rule

Write in plain technical language.
Avoid marketing language.
Avoid hype.
Avoid claims like production ready, comprehensive, enterprise grade, never before done, best in class.

Do not add slogans or sentimental lines.
Do not add phrases like made with love, for the community, crafted with passion.

# Demonstration disclaimer

This repository shows Databricks capabilities.
This code serves as a demonstration.
This code does not serve as a production solution.
Users must adapt it for their environment, security posture, and controls.
# .cursor/rules.md add this block

Tone rule
Avoid marketing tone.
Avoid hype.
Avoid claims of production readiness.
Avoid claims of completeness.
Avoid sentimental slogans.

Disclaimer rule
Include the demonstration disclaimer in README.md.
Keep the disclaimer short.
Keep it factual.
Use this README. Put it at template/{{ repo_slug }}/README.md.

# {{ project_name }}

This repository shows Databricks capabilities through a focused demonstration.
This repository does not provide a production solution.

This project targets Ubuntu Linux.
This project targets Python 3.12 or newer.

Run local checks.
./scripts/check.sh
Now enforce Linux and Python 3.12 in scripts/check.sh. Put this at the top of the script.

python - <<'PY'
import platform, sys

if platform.system() != "Linux":
    raise SystemExit("This repo supports Ubuntu Linux only")

if sys.version_info < (3, 12):
    raise SystemExit("This repo requires Python 3.12 or newer")

print("Platform OK")
print("Python OK")
PY
Update pyproject.toml.j2 to hard set 3.12.

[project]
name = "{{ project_name }}"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = []

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "SIM", "RUF"]
ignore = ["E501"]

[tool.ruff.format]
quote-style = "double"

[tool.mypy]
python_version = "3.12"
strict = true
warn_unused_ignores = true
no_implicit_optional = true

[tool.pytest.ini_options]
testpaths = ["tests"]
That set stops Mac and Windows drift. It stops Python version drift. It stops hype. It adds the Databricks demo disclaimer. It keeps tone technical.


You said:
I also want to have something that ensures all the demos use batabricks best practices and including services like genie 


ChatGPT said:


Yes. Enforce “Databricks demo best practices” as a first class rule. Do it in three layers. A contract rule. A required project checklist file. A local gate that fails when the checklist stays incomplete.

Add this section to CONTRACT.md.

# Databricks demo best practices rule

Every demo must follow Databricks best practices.

Every demo must use Unity Catalog for governed data objects.
Every demo must define clear layers for data tables.
Every demo must show lineage or provenance through named tables and repeatable steps.

Every demo must include a natural language consumption path through AI/BI Genie when the demo targets business users.
Genie setup must follow the Genie curation guidance.

Demos must deploy through Databricks Asset Bundles when the demo creates jobs, pipelines, or apps.
This aligns with Databricks guidance on Asset Bundles, Unity Catalog best practices, and Genie space curation. 

Now add a required file per project. Put it next to the project brief.

docs/projects/<PROJECT_ID>/DATABRICKS.md

Use this template.

# Databricks alignment

## Workspace and governance
Unity Catalog in use:
Catalog and schema:
Permissions model:
Lineage expectation:

## Data objects
Bronze tables:
Silver tables:
Gold tables:
Primary keys declared:
Foreign keys declared:

## Pipelines and orchestration
Asset Bundle in use:
Jobs or pipelines:
Parameters:

## Serving and consumption
Genie space created:
Curated tables added:
Joins configured:
Metrics definition:
Feedback loop:

## Operational notes
Cost guardrails:
Failure modes:
Runbook link:
The Genie section matters because Genie relies on curated context and benefits from defined relationships and governance in Unity Catalog. 

Then enforce the existence and minimum completeness. Add this block to scripts/check.sh.

python - <<'PY'
import sys
from pathlib import Path

root = Path("docs/projects")
if not root.exists():
    sys.exit(0)

required = {"PROJECT.md","DATA.md","MATH.md","LOGIC.md","REPRO.md","DATABRICKS.md"}

bad = []
for d in root.glob("*"):
    if not d.is_dir():
        continue
    present = {p.name for p in d.glob("*.md")}
    missing = sorted(required - present)
    if missing:
        bad.append((str(d), missing))

if bad:
    print("Project docs incomplete. Each project needs PROJECT, DATA, MATH, LOGIC, REPRO, DATABRICKS.")
    for folder, missing in bad:
        print(folder + " missing " + ", ".join(missing))
    sys.exit(1)

for d in root.glob("*"):
    if not d.is_dir():
        continue
    p = d / "DATABRICKS.md"
    txt = p.read_text(encoding="utf-8", errors="ignore")
    must = [
        "Unity Catalog in use:",
        "Asset Bundle in use:",
        "Genie space created:",
    ]
    if any(m not in txt for m in must):
        print("DATABRICKS.md missing required fields in " + str(d))
        sys.exit(1)
PY
Finally, make Cursor stop “demo theater.” Add this to .cursor/rules.md.

Databricks best practices rule
When you produce a demo, you must include Unity Catalog alignment, Asset Bundle deployment, and Genie readiness when the demo targets business users.
You must write docs/projects/<PROJECT_ID>/DATABRICKS.md.
You must not claim production readiness.
You must not claim completeness.
If you want an even harder gate, require a bundle file whenever a demo includes jobs or apps. Databricks Asset Bundles exist for this exact purpose. 


You said:
Ok I want it to make an DAB 

ChatGPT said:
Yes. Make every demo ship as a Databricks Asset Bundle.

Use this folder in each repo.

bundle/
  databricks.yml
  resources/
    jobs.yml
    pipelines.yml
  notebooks/
    01_ingest.py
    02_transform.py
    03_gold.py
Start with this bundle/databricks.yml.

bundle:
  name: demo_bundle

include:
  - resources/*.yml

workspace:
  host: ${DATABRICKS_HOST}

targets:
  dev:
    default: true
    workspace:
      root_path: /Workspace/Shared/demos/${bundle.name}

  prod:
    workspace:
      root_path: /Workspace/Shared/demos/${bundle.name}
Add a job in bundle/resources/jobs.yml.

resources:
  jobs:
    demo_job:
      name: demo_job
      tasks:
        - task_key: ingest
          notebook_task:
            notebook_path: ../notebooks/01_ingest.py

        - task_key: transform
          depends_on:
            - task_key: ingest
          notebook_task:
            notebook_path: ../notebooks/02_transform.py

        - task_key: gold
          depends_on:
            - task_key: transform
          notebook_task:
            notebook_path: ../notebooks/03_gold.py
Add a Delta Live Tables pipeline in bundle/resources/pipelines.yml if you want DLT. If you do not want DLT, skip this file.

resources:
  pipelines:
    demo_pipeline:
      name: demo_pipeline
      edition: core
      channel: CURRENT
      continuous: false
      development: true
      catalog: ${var.uc_catalog}
      target: ${var.uc_schema}
      libraries:
        - notebook:
            path: ../notebooks/02_transform.py
Add bundle variables in bundle/databricks.yml under targets. Keep the catalog and schema explicit.

variables:
  uc_catalog:
    default: main
  uc_schema:
    default: demo
Now make the notebooks write governed tables in Unity Catalog. Use three layers. Use bronze, silver, gold table names. Keep it simple.

bundle/notebooks/01_ingest.py

from __future__ import annotations

import logging
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)

CATALOG = dbutils.widgets.get("uc_catalog") if "dbutils" in globals() else "main"
SCHEMA = dbutils.widgets.get("uc_schema") if "dbutils" in globals() else "demo"

BRONZE = f"{CATALOG}.{SCHEMA}.bronze_events"

def main() -> None:
    """Load raw source data and write a bronze table."""
    df = spark.range(0, 1000).withColumn("event_ts", F.current_timestamp())
    logger.info("Writing bronze table %s with %s rows", BRONZE, df.count())
    spark.sql(f"create schema if not exists {CATALOG}.{SCHEMA}")
    df.write.mode("overwrite").saveAsTable(BRONZE)

main()
bundle/notebooks/02_transform.py

from __future__ import annotations

import logging
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)

CATALOG = dbutils.widgets.get("uc_catalog") if "dbutils" in globals() else "main"
SCHEMA = dbutils.widgets.get("uc_schema") if "dbutils" in globals() else "demo"

BRONZE = f"{CATALOG}.{SCHEMA}.bronze_events"
SILVER = f"{CATALOG}.{SCHEMA}.silver_events"

def main() -> None:
    """Transform bronze to silver with stable keys and normalized fields."""
    df = spark.table(BRONZE).withColumn("event_date", F.to_date("event_ts"))
    df = df.withColumn("event_id", F.sha2(F.concat_ws("|", F.col("id"), F.col("event_ts")), 256))
    logger.info("Writing silver table %s", SILVER)
    df.write.mode("overwrite").saveAsTable(SILVER)

main()
bundle/notebooks/03_gold.py

from __future__ import annotations

import logging
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)

CATALOG = dbutils.widgets.get("uc_catalog") if "dbutils" in globals() else "main"
SCHEMA = dbutils.widgets.get("uc_schema") if "dbutils" in globals() else "demo"

SILVER = f"{CATALOG}.{SCHEMA}.silver_events"
GOLD = f"{CATALOG}.{SCHEMA}.gold_daily_counts"

def main() -> None:
    """Create a gold table that supports BI queries."""
    df = spark.table(SILVER)
    out = df.groupBy("event_date").agg(F.count("*").alias("event_count"))
    logger.info("Writing gold table %s", GOLD)
    out.write.mode("overwrite").saveAsTable(GOLD)

main()
On Ubuntu, install the Databricks CLI and deploy.

pipx install databricks-cli
databricks auth login --host "$DATABRICKS_HOST"
cd bundle
databricks bundle validate
databricks bundle deploy -t dev
databricks bundle run demo_job -t dev
Genie does not deploy through DAB today in a clean, reliable way. Treat Genie as a required manual step for business facing demos. Keep it in docs/projects/<PROJECT_ID>/DATABRICKS.md. State the Genie space name and the curated gold tables.

If you want, I can turn this into your template defaults. I will add the bundle folder, wire variables, and add a DATABRICKS.md section that requires DAB plus Genie readiness.


You said:
I want to use AI dev kit 


GitHub - databricks-solutions/ai-dev-kit: Databricks Toolkit for Coding Agents provided by Field Engineering https://share.google/PbuEKkhGZLmIjc625

Databricks apps should follow guidance from 
Databricks Apps Cookbook https://share.google/pxvXmbm7OWdPToi7w

ChatGPT said:

Thought for 8s
Yes. You can wire both into your contract and your template. You can also add a gate so Cursor must anchor demos on AI Dev Kit patterns and Apps Cookbook patterns.

You want three moves.

You pin the sources. You make a required checklist per demo. You make checks fail when the checklist stays empty.

AI Dev Kit gives coding agents trusted Databricks patterns and “skills.” 

Apps Cookbook gives working examples and deployment guidance for Databricks Apps. 

Databricks docs also publish best practices for Apps. 

Add to CONTRACT.md
Paste this as a new section.

# Databricks demo alignment

Every demo follows Databricks best practices.

Every demo uses Unity Catalog for governed tables and volumes.
Every demo uses Databricks Asset Bundles for deployment when it creates jobs, pipelines, or apps.
Every demo uses AI Dev Kit patterns for agent oriented work and Databricks APIs.
Every Databricks App demo follows Apps Cookbook conventions and Databricks Apps best practices.

If the demo targets business users, the demo includes a Genie path.
The demo documents the Genie space, curated tables, and metrics.
Add a required file per project
Add this file next to your project brief.

docs/projects/<PROJECT_ID>/DATABRICKS.md

Use this template.

# Databricks alignment

## Reference sources
AI Dev Kit reference:
Apps Cookbook reference:
Databricks Apps best practices reference:

## Governance
Unity Catalog in use:
Catalog:
Schema:
Volumes:
Grants:

## Deployment
Asset Bundle in use:
Bundle path:
Targets:
Parameters:

## App pattern
App framework:
App entrypoint:
Auth pattern:
Secrets pattern:
Performance notes:

## Genie path
Genie space:
Curated tables:
Defined metrics:
Join guidance:
Prompt guidance:

## Verification
Checklist complete:
Reviewer:
Date:
Cursor must fill this. No placeholders.

Add template defaults that point to the sources
Add a file under docs/memory/ so every repo records the standard sources once.

docs/memory/DATABRICKS_SOURCES.md

# Databricks sources

AI Dev Kit
https://github.com/databricks-solutions/ai-dev-kit

Databricks Apps Cookbook
https://github.com/databricks-solutions/databricks-apps-cookbook
https://apps-cookbook.dev

Databricks Apps best practices
https://docs.databricks.com/aws/en/dev-tools/databricks-apps/best-practices
Those links stay durable. They keep you from schlepping code while still preserving “how we do it here.” 

Enforce it in scripts/check.sh
Add this block. It fails when a project exists without DATABRICKS.md, or when the file stays empty.

python - <<'PY'
import sys
from pathlib import Path

root = Path("docs/projects")
if not root.exists():
    sys.exit(0)

required = {"PROJECT.md","DATA.md","MATH.md","LOGIC.md","REPRO.md","DATABRICKS.md"}

for d in root.glob("*"):
    if not d.is_dir():
        continue
    present = {p.name for p in d.glob("*.md")}
    missing = sorted(required - present)
    if missing:
        print("Project docs incomplete in " + str(d))
        print("Missing " + ", ".join(missing))
        sys.exit(1)

    p = d / "DATABRICKS.md"
    txt = p.read_text(encoding="utf-8", errors="ignore").strip()
    if len(txt) < 200:
        print("DATABRICKS.md too small in " + str(d))
        sys.exit(1)

    must = [
        "AI Dev Kit reference:",
        "Apps Cookbook reference:",
        "Databricks Apps best practices reference:",
        "Unity Catalog in use:",
        "Asset Bundle in use:",
    ]
    if any(m not in txt for m in must):
        print("DATABRICKS.md missing required fields in " + str(d))
        sys.exit(1)
PY
Add Cursor rules
Add this to .cursor/rules.md.

Databricks alignment rule

When you build or modify a demo, you must update docs/projects/<PROJECT_ID>/DATABRICKS.md.

You must ground app patterns in Apps Cookbook and Databricks Apps best practices.
You must ground agent patterns in AI Dev Kit.

You must not claim production readiness.
You must not claim comprehensive coverage.
You must keep tone technical and specific.
Optional but useful
If you plan to run coding agents through Databricks controls, Databricks has a coding agent integration feature built on AI Gateway. That can help with governance and usage tracking. 


You said:
How do I add that 

ChatGPT said:
You add AI Dev Kit and Apps Cookbook in three concrete layers:

Reference layer

Template layer

Enforcement layer

I will walk you through exact steps inside your template repo.

Step 1. Add AI Dev Kit as a Reference Source
Do not vendor the whole repo. Do not copy large chunks.

Inside your template, add:

docs/memory/DATABRICKS_SOURCES.md

# Databricks Standards

AI Dev Kit
https://github.com/databricks-solutions/ai-dev-kit

Databricks Apps Cookbook
https://github.com/databricks-solutions/databricks-apps-cookbook
https://apps-cookbook.dev

Databricks Apps best practices
https://docs.databricks.com/aws/en/dev-tools/databricks-apps/best-practices
This makes the source of truth durable.

Step 2. Add AI Dev Kit as a Git Submodule (Optional but Strong)
If you want agents to reference the code patterns directly:

git submodule add https://github.com/databricks-solutions/ai-dev-kit.git external/ai-dev-kit
git commit -m "Add AI Dev Kit reference"
Then document in README:

To update AI Dev Kit reference:

git submodule update --remote external/ai-dev-kit
This keeps the toolkit available without polluting your demo logic.

Step 3. Add Apps Cookbook Pattern Enforcement
Add a required file per project.

docs/projects/<PROJECT_ID>/DATABRICKS.md

Make this part of the template.

Now enforce it in scripts/check.sh.

Add this block:

python - <<'PY'
import sys
from pathlib import Path

root = Path("docs/projects")
if not root.exists():
    sys.exit(0)

required = {"PROJECT.md","DATA.md","MATH.md","LOGIC.md","REPRO.md","DATABRICKS.md"}

for d in root.glob("*"):
    if not d.is_dir():
        continue
    present = {p.name for p in d.glob("*.md")}
    missing = sorted(required - present)
    if missing:
        print("Project docs incomplete in " + str(d))
        print("Missing " + ", ".join(missing))
        sys.exit(1)

    txt = (d / "DATABRICKS.md").read_text(encoding="utf-8", errors="ignore")

    required_fields = [
        "AI Dev Kit reference:",
        "Apps Cookbook reference:",
        "Unity Catalog in use:",
        "Asset Bundle in use:",
    ]

    for field in required_fields:
        if field not in txt:
            print("Missing Databricks alignment field in " + str(d))
            sys.exit(1)
PY
Now demos fail CI unless they explicitly align.

Step 4. Add Bundle Requirement Enforcement
Inside scripts/check.sh, enforce DAB presence when bundle exists:

if [ -d "bundle" ]; then
  if [ ! -f "bundle/databricks.yml" ]; then
    echo "Bundle folder exists but databricks.yml missing"
    exit 1
  fi
fi
If a demo creates jobs or apps, the bundle must exist.

Step 5. Prevent Marketing Drift
Add this block to .cursor/rules.md:

Databricks demo tone rule

When describing demos:
Use precise technical language.
Avoid claims like production ready, comprehensive, enterprise grade, never before done.

State that this is a demonstration of Databricks capabilities.
Do not claim this is a full production solution.
Then enforce it in scripts/check.sh:

python - <<'PY'
import re, sys
from pathlib import Path

deny_phrases = [
    "production ready",
    "enterprise grade",
    "comprehensive solution",
    "never before",
    "best in class",
    "made with love",
]

bad = []
for p in Path(".").rglob("*.md"):
    txt = p.read_text(encoding="utf-8", errors="ignore").lower()
    for phrase in deny_phrases:
        if phrase in txt:
            bad.append((str(p), phrase))

if bad:
    print("Marketing language detected:")
    for f, phrase in bad:
        print(f" - {f} contains '{phrase}'")
    sys.exit(1)
PY
Step 6. Add a Simple AI Dev Kit Usage Contract
Inside CONTRACT.md, add:

# AI Dev Kit alignment

Agent workflows must follow patterns from AI Dev Kit.
Do not invent new abstractions if AI Dev Kit provides one.
Reuse validated patterns.
Document any deviation in docs/memory/DECISIONS.md.
What You Now Have
Every demo:

Uses Unity Catalog.
Uses DAB for deployment.
Documents Genie path if relevant.
References AI Dev Kit patterns.
Follows Apps Cookbook structure.
Avoids marketing claims.
Fails CI if alignment is missing.

This gives you technical discipline instead of demo theater.

If you want, next I can give you a hardened minimal Databricks App template that follows Apps Cookbook structure and plugs into your Copier template automatically.






