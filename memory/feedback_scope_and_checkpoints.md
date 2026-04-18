---
name: User preferences — scope, checkpoints, ablation flags
description: Keep scope minimal; preserve all ablation flags; no periodic checkpoints
type: feedback
originSessionId: 2be47fb7-b67d-4054-9d23-8841007fd70f
---
Three validated preferences from this user:

1. **Preserve ablation flags even in "strict reproduction" code.** Flags like `--use_rot6d`, `--use_upper_body`, `--use_expression`, `--root_normalize`, `--use_phono_attribute` on `train_NeuralSignActors.py` are intentional — user runs both paper-default AND ablated variants from the same script. Do not suggest removing them.

2. **Only `newest_model.pt` + `best_model.pt` — no periodic checkpoints.** User explicitly rejected adding "every N epoch" checkpoint saves.

3. **Small, surgical fixes over broad refactors.** When given a bug list, user picks specific items to fix (e.g. "fix item 1 only") rather than approving bulk cleanup. Confirm scope before touching more than the requested item.

**Why:** User is running a research codebase with multiple parallel experiments (paper reproduction + own method); flexibility and minimal disk usage matter more than tidiness.

**How to apply:** When reviewing code, flag issues but wait for explicit approval before fixing. Never remove CLI flags or config options unprompted. Default to asking "要我改哪些?" when multiple fixes are possible.
