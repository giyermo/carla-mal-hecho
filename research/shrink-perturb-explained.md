# Shrink & Perturb: what it does and why (for the defence)

*Written 1 November 2026, from a full read of [Ash & Adams, "On Warm-Starting Neural Network Training", NeurIPS 2020](https://arxiv.org/abs/1910.08475) (`papers/warm_start.pdf`, arXiv v3). Section numbers below refer to that paper. Companion to `experiment-02-warm-start.md`, which is the experiment; this file is the method.*

> Everything attributed to a section of the paper was read there. Anything marked **my reasoning** is not in the paper and needs to be defended as my own argument, or dropped.

---

## 1. The problem it solves

Data arrive piecemeal; you keep rebuilding a model on a growing dataset. The obvious move is to start each round from the previous round's weights. For convex problems that is standard and works (§1). For neural networks it **damages generalization while leaving training accuracy untouched** (§2): the warm-started model fits the training data just as well and tests worse.

The paper's headline evidence (§2.1, Table 1), ResNet-18 on CIFAR-10, trained on 50% then on 100%:

| | SGD | Adam |
|---|---|---|
| Random init | 56.2 (1.0) | 78.0 (0.6) |
| Warm start | 51.7 (0.9) | 74.4 (0.9) |

Validation accuracy %, std over runs in brackets. The same gap appears on CIFAR-100 (18.2 → 15.5) and, much smaller, on SVHN (89.4 → 87.5), which the paper attributes to SVHN being an easier problem. **Logistic regression is not damaged at all** — the loss surface is convex — which is the cleanest sign that this is about non-convex optimization, not about data (§2.1).

Three more findings that matter for how we run our rounds:

- **The gap grows as the stream goes on** (§2.2, Figure 2): with CIFAR-10 fed 1,000 samples at a time, the warm/random validation gap *widens* with more rounds, while warm start keeps its large wall-clock advantage.
- **Damage is immediate** (§3.2, Figure 4): warm-starting from a checkpoint trained only a few epochs on the incomplete data already hurts. You cannot avoid it by stopping the first round early.
- **The gap is inversely proportional to how much data the first round had** (§2.1, citing Appendix Figure 10). So the damage is worst when round 1 is small relative to what follows.

And one that kills the obvious workaround: **hyperparameters can close the gap, but only by throwing away the speed-up** (§3.1, Figure 3). Sweeping batch sizes {16, 32, 64, 128} × learning rates {0.001, 0.01, 0.1}, the warm-started models that generalize as well as random ones take about as long to train as random ones. The paper's explanation (§3.1, Appendix Figure 11): those models have little weight correlation with their initialization, i.e. they **forgot** round 1 — which is why they cost the same. Regularization does not fix it either (§3.3): L2, confidence penalties and adversarial training all help both initializations about equally, so the gap survives.

## 2. The trick

At round *t*, initialize each learnable parameter from the previous round's as

> θᵢᵗ ← λ·θᵢᵗ⁻¹ + pᵗ,  pᵗ ~ N(0, σ²),  0 < λ < 1  (§4)

with one practical substitution the paper makes explicitly (§4, the paragraph on the perturbation step):

> "In practice, we add noise by adding parameters from a scaled, randomly-initialized network, to compensate for the fact that many random initialization schemes use different variances for different kinds of parameters."

So the noise is not one Gaussian with a single σ across the whole network: it is `noise_scale × θ_random`, where `θ_random` is a freshly initialized network of the same architecture. **This is exactly what `experiments/warm_start/shrink_perturb.py` does**, so our implementation follows the paper's real recipe rather than its abbreviated formula.

Two parts, two jobs.

### Shrinking: keeps the hypothesis, raises the loss

For a bias-free, BatchNorm-free ReLU network, scaling every weight by λ scales the pre-softmax outputs by λᴸ for depth *L*, which leaves `arg max fθ(x)` unchanged (§4, Proposition 1 in the appendix). The *decisions* survive; the *confidence* drops, because the softmax distribution moves toward uniform and the cross-entropy loss goes up.

For real architectures the preservation still roughly holds, for a reason worth knowing: **BatchNorm's running mean and variance compensate for the rescaling**, so in a ResNet "only extreme amounts of shrinking are able to damage classifier performance" (§4, Figure 6). For a ReLU MLP *with* bias nodes, damage only appears below λ ≈ 0.6.

### Perturbing: mostly for speed

Noise "improves both training time and generalization performance", and in the λ × noise-scale grid (§4, Figure 8) adding noise mainly makes models train *faster*, with a smaller effect on accuracy. Noise alone is not a substitute: adding a lot of parameter noise without shrinking "damages the learned function", drastically increasing training time without closing the gap (§4, citing Appendix Table 4).

## 3. Why it works: gradient balance

This is the paper's actual explanation (§4, Figure 5), and it is the part to have ready for a question.

In a warm-started model, the gradients from **new, unseen** samples are much larger in magnitude than those from samples it already fits. The model is already at a minimum for the old data, so the old data contribute almost nothing; the new data dominate. The paper connects this imbalance to the known difficulty of imbalanced gradient contributions in multi-task learning, and argues that under warm starting "the model does not learn in the same way as it would with randomly-initialized training."

Shrinking the weights **raises the loss on data the model already fits**, which restores the magnitude of their gradients. Figure 5 shows the two gradient norms — old half and new half of the data — being brought back into balance by Shrink & Perturb, while the model still fits the training data far faster than a random initialization. The paper's summary: the trick's success "may lie in its ability to standardize gradients while preserving a model's learned hypothesis."

This also explains why the alternatives fail (§4): shrinking only the last layer (Appendix Figure 14), or penalizing confidence during training (Appendix Table 3), both reduce confidence without rebalancing gradients throughout the network. "For sophisticated architectures especially, we find it is important to holistically modify parameters before training on new data."

**My reasoning, not in the paper:** in a BatchNorm network the loss-raising story needs care. BN makes conv weights scale-invariant, so shrinking them barely changes the function — which the paper itself says (Figure 6). What *does* change is the ratio between gradient size and weight size, i.e. the effective learning rate: smaller weights mean larger relative steps, so plasticity returns. That is the standard "effective learning rate" argument for normalized networks, but **I have not verified a citation for it** — find one before putting it in the thesis, or present it as an observation.

**Also my reasoning:** the entropy half of the argument does not transfer cleanly to our case. YOLO's loss is mostly box regression (CIoU + DFL), and with a single class the classification term is nearly degenerate — there is no softmax over classes whose entropy shrinking could raise. So if Shrink & Perturb helps a one-class detector, it is via the gradient-balance and effective-learning-rate mechanisms, not via confidence. This is a real difference between their setting and ours and worth saying out loud in the thesis.

## 4. The settings we use, and where they come from

`λ = 0.6`, `noise_scale = 0.01`. That is the paper's headline configuration: Figure 7 fixes the noise scale at 0.01 and varies λ, and reports that **λ = 0.6 "performs identically to randomly initializing in terms of validation accuracy, but trains much more quickly"** (§4). λ = 1 is pure warm start, λ = 0 pure random init. Figure 8's grid adds that accuracy "tends to improve with more aggressive shrinking" while train time rises as λ falls, and the paper tempers this with: "one does not need to shrink very aggressively to adequately enough correct gradients and efficiently close the warm-start generalization gap."

Note the paper uses a *different* setting for its transfer-learning experiment (§4.2, Figure 9): λ = 0.3, noise scale 0.0001. So the best λ is not universal across settings in their own work either — which is the justification for our λ ∈ {0.3, 0.6, 0.9} sweep at round 4.

§4.1 is worth knowing to pre-empt a question: applying the trick at *every* SGD step is close to an aggressive noisy L2 regularizer, and gives only a marginal improvement on a static dataset. The regularization benefit is real but small and is **not** the explanation for the main result — L2-regularized models still suffer the warm-start gap (Appendix Table 3). We apply it once per round, which is the paper's main setting and the one that matches "a new flight arrived."

## 5. How our implementation makes each decision

`experiments/warm_start/shrink_perturb.py`, applied to round *k−1*'s `last.pt` before round *k*:

| Decision | What we do | Why |
|---|---|---|
| Which tensors | Everything in `named_parameters()` | The paper modifies every learnable parameter and argues explicitly against partial (last-layer-only) modification |
| BatchNorm running stats | **Left unchanged** (they are buffers, not parameters) | They are data statistics, not gradient-learned weights. The paper's formula covers learnable parameters only. *Caveat:* right after shrinking they are stale relative to the new activations; YOLO's momentum updates will correct them over the first batches |
| Noise source | A freshly initialized `DetectionModel` of the same architecture, scaled by 0.01 | The paper's own practice, so each layer gets noise matched to that layer's init variance |
| Noise seed | `seed × 100 + round` | Reproducible, but different noise each round |
| Starting checkpoint | `last.pt`, not `best.pt` | `best.pt` is chosen on the validation set; using it would mix model selection into the comparison |
| Applied | Once per round | §4.1: every-step application is a different thing (≈ noisy weight decay) |

Verified when the script was written: weights come out multiplied by exactly 0.600, BN statistics are untouched, and Ultralytics reloads 708/708 items.

## 6. Why round 4 is the round that matters

§4.2 is the section that fits our round 4, and it is not the section people usually quote. It warm-starts a ResNet from **one dataset onto a different but related one**, at varying fractions of target data, and finds:

- when the target data are few, warm starting (i.e. pre-training) helps;
- as the target data grow, there is "frequently a crossover point" past which training from scratch generalizes better;
- when source and target are dissimilar, random init can be better regardless of target size;
- the crossover "cannot be easily predicted" from the model or the data;
- and **Shrink & Perturb performs at least as well as whichever of the two is better**, so you do not have to predict it.

That is our situation almost exactly. The high flight is a different-but-related dataset (824 tiles) arriving on top of an archive of low flights (1864 tiles), i.e. the new condition is 31% of round 4's data. Rounds 1–3 were the *stationary* case the rest of the paper studies, which is why the three setups came out within one mAP50 point of each other: there was no shift for the restart rule to get right.

So the prediction for round 4, stated before the numbers come in (**my reasoning**, by analogy with §4.2 and §2.2):

- `warm` should be the weakest on `high_1080p` — it sits in a low-altitude solution, the new flight's gradients dominate but the initialization resists moving, and this is exactly the "warm start under domain shift" case §4.2 says deteriorates;
- `cold` should be strong on both test sets but is the expensive reference;
- `sp` should be at least as good as the better of the two, per §4.2's claim, and the interesting question is whether it gets there at 10 epochs instead of 30;
- a **lower λ** should do better at round 4 than at rounds 2–3, because more shrinking means more plasticity, and the paper used λ = 0.3 for its own shift experiment.

## 7. Differences between their setup and ours

Keep this list; it is the first thing an examiner will probe.

| | Ash & Adams | Ours |
|---|---|---|
| Task | Image classification (softmax, accuracy) | One-class object detection (box regression + DFL, mAP50) |
| Models | ResNet-18, 3-layer MLP, logistic regression | YOLO26n, 2.4 M parameters |
| Data | CIFAR-10/100, SVHN | 640 px tiles from DJI drone video |
| Augmentation | **None** (§3, stated explicitly) | YOLO's full default pipeline (mosaic, HSV, flips, …) |
| LR schedule | **None** (§3, stated explicitly) | Ultralytics warmup + decay, restarted every round |
| Rounds | i.i.d. halves or random 1,000-sample batches of one dataset | Whole flights, i.e. correlated tiles and, at round 4, a genuine condition change |
| Training length | To convergence each round | Fixed 30 epochs (10 for the `ep10` variant) |
| First round | Often 50% of the data | Round 1 is 704 of 2688 tiles (26%) |

The augmentation and LR-schedule rows are the most important: both are known to improve generalization, and YOLO restarts its LR warmup every round. **My reasoning:** either could already be doing part of Shrink & Perturb's job, which would explain why rounds 1–3 showed a 0.5–1 point effect instead of the paper's 3–4.5 points. If round 4 also shows a small effect, "YOLO's training recipe already absorbs the warm-start problem" is a legitimate and publishable conclusion — it just has to be argued from the paired per-seed differences, not from overlapping error bars.

One more asymmetry: their gap is measured at *equal training length* (to convergence), so their speed claim is separate from their accuracy claim. Ours fixes the epoch count, which makes the accuracy comparison clean but means the speed benefit only shows up in the `ep10` variant. Without that variant we could not claim any saving.

## 8. What to look for in our results

From `summarize.py`'s matrix and paired tables:

1. `high_1080p` at round 4 — plasticity. `cold` is the reference; `warm` below it is the paper's effect under shift; `sp` at or above `cold` is the useful outcome.
2. `low_*` at round 4 — retention. If `warm` wins here while losing badly above, that is not warm doing well, it is warm not moving.
3. `sp_ep10` vs `cold` at 30 epochs — the only comparison in which the trick is *useful* rather than merely harmless.
4. `sp_lam0.3` vs `sp` vs `sp_lam0.9` at round 4 — whether the best shrink depends on the arriving flight being a new condition. If it does, λ becomes a decision the loop could learn (contribution D in `carla-rl-state-of-the-art.md`).
5. Signs, not means: three seeds agreeing in `paired.csv` beats a larger mean difference with mixed signs.

## 9. One-paragraph version

Warm-starting a network on a growing dataset reaches the same training loss but generalizes worse than retraining from scratch, because the old data sit at a minimum and stop producing gradients while the new data dominate. Shrink & Perturb multiplies every learnable weight by λ = 0.6 and adds 1% of a fresh random initialization: the shrink preserves what the network decided while raising its loss, so the old data produce useful gradients again and the two halves balance, and the noise mainly speeds training up. It costs one checkpoint operation, so it gives cold-restart generalization at warm-start cost — if it transfers from CIFAR classification to a one-class drone detector, which is what round 4 of our experiment is for.
