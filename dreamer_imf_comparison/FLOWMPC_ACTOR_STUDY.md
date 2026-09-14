# ReBRAC + ITPO controller diagnostic

This study replaces the earlier Dreamer-style REINFORCE actor loop with the
controller used by the ITPO/FlowMPC paper. It asks one narrow question:

> Does one gradient-ascent policy update through fixed-noise trajectory-iMF
> rollouts improve the same offline-trained ReBRAC policy over its zero-shot
> behavior on Reacher?

## Paper-matched components

- deterministic ReBRAC actor: three ReLU hidden layers of width 256;
- twin critics with ReLU followed by LayerNorm in each hidden layer;
- Adam at `1e-3`, batch size 1024, discount `0.99`, target rate `0.005`;
- actor and critic behavior penalties, target-policy smoothing, delayed actor
  updates, Q normalization, and released target-update timing;
- one million offline ReBRAC updates;
- at each real state, persistent plain gradient ascent on discounted predicted
  rewards plus the frozen terminal minimum twin-Q value;
- newly sampled Monte Carlo noise at each real state, held fixed across the
  inner gradient step;
- one tuning environment seed that is disjoint from evaluation seeds;
- a paired comparison against the exact same unadapted ReBRAC checkpoint.

## Explicit adaptation boundaries

This is not labeled a full reproduction of FlowMPC. The paper evaluates D4RL
Gym-MuJoCo with fully observed state-space models and policy-tilted MeanFlow
training. This diagnostic uses the existing 50k-transition random/smooth DMC
Reacher replay, an RSSM trajectory-iMF prior, decoded observations inside
imagined rollouts, and an already-trained non-tilted dynamics model. The paper
does not report ReBRAC coefficients or controller hyperparameters for Reacher,
so released ReBRAC coefficient defaults are used; `H=5`, `E=1`, and `M=4096`
are fixed to common paper settings and only the paper's non-Hopper step-size
grid is tuned.

## Evidence contract

The run uses six fresh ReBRAC cells (three world-model datasets by two policy
seeds), one disjoint-seed tuning stage, and six paired evaluation cells with
five episodes per condition. Earlier learned actors are forbidden. Every
source checkpoint, result, trace, and marker is SHA-256 bound, and policy plus
environment replay is repeated before a cell is accepted.

The primary unit is the world-model seed. Actor seeds and episodes are nested.
The primary result is the IQM of the three paired world-seed mean-return
contrasts, FlowMPC minus zero-shot ReBRAC. This remains exploratory evidence.
