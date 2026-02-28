
CLAUDE.md
Claude reads this well. Cursor also tends to follow it.

You are a code assistant in this repo. Obey CONTRACT.md.

When you start work, do this.

Read CONTRACT.md.
State the files you will touch.
State the local checks you will run.

When you propose changes, do this.

Show a short plan.
Make small diffs.
Keep modules focused.
Avoid a catch-all util module.

Do not add dependencies unless I approve.
Do not change schemas unless you present a migration plan.

Before you finish, do this.

Run local checks via scripts/check.sh.
Fix every failure.
Do not suggest a push until checks pass.