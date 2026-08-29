# Checkpoints from before the movement-speed correction

These ran on the engine as it was until 2026-08-30, where every unit moved 20%
slower than the game does. See `docs/SPEED_SCALE_CORRECTION.md`.

They are kept because they are the only evidence about the anchor coefficient
that exists, and because throwing away a checkpoint that scored 31-9 without
having anything to replace it would be careless. They are **not** candidates
for ladder: they are tuned to physics the simulator no longer has.

| run | anchor-coef | best step | record | hog share |
|---|---|---|---|---|
| anchor_cf005 | 0.05 | 1,452,032 | 25-15 | 0.191 |
| anchor_cf025 | 0.25 | 1,702,400 | 31-9 | 0.182 |

Both were stopped at ~1.89M of 3M steps when the speed correction landed,
because finishing them would have spent three more hours training against the
wrong game. On this evidence the stronger anchor is the better one, and that is
the question the two runs launched afterwards ask again on correct physics.
