# The Fluid in the Wave Function — companion code

**Article:** [Quest for Entropy #9 — "The Fluid in the Wave Function"](https://questforentropy.substack.com/p/the-fluid-in-the-wave-function)

**Series:** ← [#8 The Crypto Bet](https://github.com/masteris777/quest-for-entropy-the-crypto-bet) · [#10 The Almost-Crystal](https://github.com/masteris777/quest-for-entropy-the-almost-crystal) →

Everything the article quotes, runnable from scratch.

## Run it

```
pip install numpy scipy matplotlib
python run_all.py
```

`run_all.py` runs every experiment and checks each number the article prints —
eleven checks, about a minute on a laptop. It reports PASS or FAIL per claim.
The lattice ħ hunt is the slow part; `python run_all.py --quick` skips it.

## What is in here

| file | what it does |
|---|---|
| `the_zip.py` | the three-panel animation: one wave function read as two curves, a density and a flow |
| `the_letter.py` | the same packet with and without the *i*, as a space-time picture |
| `pressure_slot_test.py` | fires three packets through one medium and measures what sits in the pressure slot |
| `hbar_from_the_lattice.py` | extracts an effective ħ from the pendulum lattice at three grid spacings |
| `lattice_engine.py` | the lattice integrator the ħ hunt runs on |
| `exchange_rate.py` | the classical machine whose observations evolve exactly as the quantum oscillator does |
| `exchange_rate_harder.py` | the same construction for a non-integrable system |
| `early_toy_orbitals.py` | early toy: standing patterns that look like orbitals |
| `early_toy_decoherence.py` | early toy: a decay curve |
| `early_toy_collapse.py` | early toy: interference washing out |

The three `early_toy_*` scripts are shipped **as they were, mistakes intact**.
The article explains what each of them does not establish; that is the point of
including them.

`article.md` is the piece itself, with its figures in `assets/`. Figures the
scripts generate land in `figures/`.

## Scope

The article's "What this does NOT claim" section is the scope fence, and it
ships with the code in `article.md`. Short version: this is a report on small
home-built engines, not a verdict on anyone's research programme.

## Licence

MIT for the code. The article text is © Marijus Masteika.
