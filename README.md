# CDD — Critique-Driven Development

> **Status: pre-alpha.** API is unstable. Expect breaking changes until v1.0.

A methodology for AI-assisted creative work where the human is editorial — not generative.

## The idea

There are four ways to produce a creative artifact at scale:

| | Human role | AI role | Logged? |
|---|---|---|---|
| **Hand-authoring** | Generates | None | N/A |
| **Procedural generation** | Defines rules | None | Rules are the log |
| **Vibes-based prompting** | One-shot prompt | Generates | No |
| **Critique-Driven Development** | Reviews each iteration | Generates at volume | **Yes — every iteration** |

CDD is the fourth quadrant. The AI proposes; the human disposes. Every prompt, seed, model identity, output, and human verdict is recorded in a log. The artifact is reproducible from the log alone.

## Why it matters

- **Authorship.** The review trail is what makes the output yours. You didn't generate the pixels; you directed the iteration that shaped them. The same way a film director didn't operate the camera but owns the film.
- **Reproducibility.** Replay the log, regenerate the artifact. (Subject to model availability — see the determinism contract below.)
- **Scale.** You can't hand-author 10,000 grass blades or 1,000 dialogue lines. You can review the rules that generate them, and reject the ones that miss.
- **Defensibility.** When someone asks "did you make this or did the AI?", the honest answer is "I directed it through N rounds of critique. Here's the log."

## Roles

CDD describes a relationship between three entities:

- **Generator** — typically AI. Produces drafts at volume.
- **Critic** — the human. Reviews each draft, issues a verdict (accept / reject / refine), and provides direction for the next iteration if needed.
- **Log** — the system. Captures the authorship trail in a structured, replay-capable format.

The critic's role is **demanding**, not passive. CDD is not "review and rubber-stamp" — it's "review with informed judgment that drives the next iteration." The log tracks what was rejected and why, not just what was accepted.

## Determinism contract

CDD adapters declare which determinism tier they target:

- **Tier A — Strict.** Replay produces byte-identical output. Requires fully recorded model identity, version, seed, temperature, all retries.
- **Tier B — Semantic.** Replay produces semantically-equivalent output. Re-rolls allowed; human re-verifies. **Default tier.**
- **Tier C — Best-effort.** Replay attempts to recreate; mismatches are logged. Useful for archival.

Tier A is brittle when AI providers change models silently. Tier B keeps the discipline alive across model deprecations.

## Installation

> **Pre-alpha — no PyPI release yet.**

```bash
git clone https://github.com/srbryers/cdd ~/Development/cdd
pip install -e ~/Development/cdd
```

When stable:

```bash
pip install cdd-loop  # PyPI name (planned)
import cdd            # import name
```

## Status

| | |
|---|---|
| Version | 0.1.0 |
| Stability | Pre-alpha |
| License | MIT |
| Python | >=3.11 |

## Background

CDD originated as the authoring discipline for [FATHOMS](https://github.com/srbryers/fathoms-game), a Caribbean pirate open world game. The repository will dogfood CDD across textures, vegetation placement, dialogue, and other content types, with issues filed here to drive the methodology forward.

## License

MIT. See [LICENSE](LICENSE).
