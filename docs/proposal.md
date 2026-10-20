# RLFIND: Reinforcement Learning with Falsifiable Inverse Nonlinear Dynamics

*Falsification is all you need*

## Abstract

Frontier AI agents now solve forward problems in theoretical physics at expert level. On TheoPhysEval, a suite of 50
unpublished research problems, the best agent answered 49 correctly. The one problem no agent solved was an inverse
problem: recovering an autonomous differential equation from a plotted trajectory. We argue this is structural.
Reinforcement learning with verifiable rewards (RLVR) works because forward answers can be checked, but inverse answers
cannot be verified. A candidate model can only be shown consistent with data or falsified by new data, and consistency
badly underdetermines the answer. Yet most of the hard steps in science are inverse: finding governing equations,
closures, the variables in which dynamics simplify, and the conditions that produce a phenomenon.

We propose RLFIND, a training paradigm in which an agent probes hidden nonlinear dynamical systems through experiments
of its own design and then submits a model together with a claimed domain of validity. An adversarial falsifier with
laboratory access to the hidden system attacks the claim. Claims that survive earn their empirical content in bits,
minus the bits needed to state the model; claims that fail are penalized in proportion to what they claimed. The reward
unifies Popper's falsifiability with Rissanen's minimum description length, contains RLVR as a degenerate special case,
and is designed against four specific failure modes of naive inverse rewards. We describe the environments, a curriculum
running from ODE identification to the structural neighborhood of the Navier–Stokes equations, an evaluation protocol
built on held-out generators and counterfactual physics, and a moonshot track aimed at unforced finite-time blowup.

**In one sentence:** reward agents for bold claims about hidden dynamical systems that survive an adversary with a lab,
minus the bits it takes to state them.

## 1. Motivation

### 1.1 Forward problems are close to saturated

TheoPhysEval (Chandra et al., 2026) gathered 50 unpublished, research-level problems from experts across theoretical
physics, each with a verifiable answer that cannot be guessed. With literature search and code execution available, the
best agent solved 49. With search disabled, the two strongest agents solved 46 each, which suggests that much of the
relevant literature is effectively memorized. The authors conclude that calculation is no longer the bottleneck; what
agents still lack is taste, and the ability to change direction when a problem turns out to be the wrong one.

Recent AI results in mathematics point the same way. The most celebrated of them, including the disproof of the unit
distance conjecture (Alon et al., 2026) and a finite-time blowup construction for the forced Navier–Stokes equations
(OpenAI, 2026), produced objects whose correctness can be checked once they are found.

### 1.2 The frontier is inverse

The steps that turn calculation into understanding mostly run in the other direction: from behavior to equations, from
fine-scale data to closures, from observations to the variables in which the dynamics simplify, and from a target
phenomenon back to the conditions that produce it. These are inverse problems, and they are typically ill-posed in
Hadamard's (1902) sense. A solution may not exist inside the hypothesis class being searched, may not be unique, and may
not depend stably on the data.

TheoPhysEval's single unsolved problem has exactly these signatures. Its authors note that a synthetically generated
inverse problem leaves the setter holding information the solver cannot recover from the prompt, that the intended
solution turns on a non-obvious reading of the prompt's own terms, and that it was the one problem flagged as resistant
to conversion into an automatically gradable number. They add that it should be possible to construct more such
problems.

### 1.3 Why RLVR does not transfer

RLVR rests on a cheap checker that separates right answers from wrong ones. For an inverse problem, the cheap checker
(simulate the candidate forward and compare with the data) separates consistent answers from inconsistent ones, which is
not the same thing. Many models fit any finite dataset. Discriminating among them takes a prior, and the prior is
precisely the science one hoped the agent would learn.

Rewarding consistency therefore teaches overfitting. Rewarding agreement with a hidden ground truth teaches the prior of
whoever generated the environments, which, at the scale RL requires, means the taste of the coding agents that wrote
them. Neither produces the capability we want.

### 1.4 A motivating example: Navier–Stokes

The breakdown alternatives in the Clay problem statement (Fefferman, 2006) permit a smooth external force. With a force
available, finite-time blowup becomes a forward problem: prescribe the singular trajectory, read off the force that
makes it a solution, and do the considerable work of making that force smooth. The unforced problem is a genuine inverse
problem: find smooth, finite-energy initial data whose own evolution concentrates into a singularity. Even the forward
check is untrustworthy there, since numerical evidence for blowup has repeatedly dissolved under refinement (see, e.g.,
Hou & Li, 2006).

The standard way to tame such a problem is a funnel: a blowup profile that attracts a finite-codimension set of initial
data, so that the dynamics itself performs the inversion. In our reading, the recent forced construction seeds each
scale of its cascade independently, which makes the blowup it builds effectively infinite-codimensional. The unforced
question is whether some cascade regenerates itself with only finitely many unstable directions. Codimension is the
right currency, and finding a funnel is an inverse problem of exactly the kind RLFIND targets.

## 2. From verification to falsification

### 2.1 Claims

An RLFIND episode ends when the agent submits a claim $(M, D, \varepsilon)$.

$M$ is a model, written as a program in a typed description language for dynamical systems. $D$ is a claimed domain of
validity: a region of the hidden system's condition space, stated in a structured language of parameter ranges, classes
of initial conditions drawn from named ensembles, time horizons, and observation channels. $\varepsilon$ assigns a
tolerance to each claimed observable.

Before submitting, the agent may run experiments on the hidden system, choosing conditions and observation channels
under a cost budget. Experimentation is what makes many inverse problems tractable at all, and learning to design
discriminating experiments is part of what we want to train.

### 2.2 The falsifier

The claim is attacked by a falsifier, a portfolio of search procedures with query access to the hidden system restricted
to $D$: random and quasi-random sampling, Bayesian optimization, gradient-based search through differentiable
simulators, and a learned adversary. For deterministic observables within the predictability horizon, a counterexample
is a condition in $D$ where the model's error exceeds its tolerance. For statistical claims, a counterexample is a
rejection by a two-sample test at significance $\alpha$, with family-wise error control across the falsifier's queries,
so that $\alpha$ bounds the probability of wrongly falsifying a true claim.

Claims come in two strengths. Typical-case claims assert that the tolerance holds for all but a small measure of
conditions in $D$, and are falsified only by a statistically significant violation on a set of positive measure, which
protects them from measure-zero pathologies. Uniform claims assert the tolerance everywhere in $D$; they carry more
content and are far easier to falsify. The agent chooses which to make.

In the language of interactive proofs (Babai, 1985), the agent is Merlin, powerful and untrusted, and the scoring core
is Arthur, weak and trusted. This Arthur, however, has a laboratory. The agent never submits predictions, only programs;
every prediction used for scoring is computed by the trusted core.

### 2.3 The reward

Let $\mathrm{Content} (D, \varepsilon)$ be the information, in bits, by which the claim narrows the observable outcomes
over $D$ relative to a naive reference forecaster such as persistence or climatology:

$$
\mathrm{Content} (D,\varepsilon) \;=\; N \int_{D} \sum_{o} \log_2^{+}\!\left (\frac{\sigma_o (x)}{\varepsilon_o (x)}\right) d\nu (x),
$$

where $o$ ranges over the claimed observables, $\sigma_o (x)$ is the reference forecaster's spread at
condition $x$, $\nu$ is a reference probability measure on the condition space, and $N$ is the effective number of
independent observations per unit measure. Let $L (M)$ be the length of the model in bits under a fixed, published
prefix-free encoding, including the definitions of any primitives the agent introduced and the precision at which each
constant is stated. With $C_{\mathrm{exp}}$ the cost of the agent's experiments, the reward is

$$
R \;=\;
\begin{cases}
\mathrm{Content} (D,\varepsilon) \;-\; L (M) \;-\; \mu\, C_{\mathrm{exp}} & \text{if the claim survives,}\\[4pt]
-\,\kappa\,\mathrm{Content} (D,\varepsilon) \;-\; \mu\, C_{\mathrm{exp}} & \text{if it is falsified.}
\end{cases}
$$

Three properties motivate this form.

First, both principal terms are measured in bits, so a surviving claim is rewarded for the net compression it achieves
on the hidden system's behavior. This is minimum description length (Rissanen, 1978), restricted to the region where the
model withstands attack.

Second, the penalty scales with what was claimed, which turns $\kappa$ into an interpretable confidence threshold.
Ignoring $L (M)$, a claim has positive expected reward only if the agent believes it survives with probability
above $\kappa/ (1+\kappa)$; setting $\kappa = 9$ means "do not claim what you are less than 90% sure of." Overclaiming
is negative in expectation, and bold claims the agent can defend are worth the most. This is Popper's (1959) preference
for theories with high empirical content, turned into an incentive.

Third, RLVR is the degenerate case. When the hidden system is a deterministic function with a unique answer, $D$ is a
single point and $\varepsilon = 0$, survival is correctness, and up to an affine transformation the reward reduces to a
binary verifiable reward.

### 2.4 Horizon-aware scoring

Chaotic systems need special handling. Beyond a few Lyapunov times, pointwise predictions decorrelate even for a perfect
model with tiny state uncertainty (Lorenz, 1963), so a trajectory-matching reward cannot tell the true model from a
wrong one. RLFIND scores trajectories only within a predictability horizon estimated from the hidden system's leading
Lyapunov exponents, and scores statistics beyond it: invariant-measure observables, spectra, correlation functions, and
extreme-value statistics, using strictly proper scoring rules (Gneiting & Raftery, 2007) and two-sample tests. Models
may be stochastic, which is the natural form for closures, and are then judged on calibration as well as sharpness.

### 2.5 An open but priced hypothesis class

The description language covers ODEs, PDEs, delay and stochastic equations, implicit relations, discrete maps, latent
variables, and nonlocal operators, and the agent may define new primitives. Every definition counts toward $L (M)$.

This has two consequences. Leaving the class a task suggests is allowed, but it must pay for itself, which addresses
problems whose answer lies outside the expected class. And data-encoding hacks defeat themselves, since a primitive that
memorizes data costs as many bits as the data it memorizes. Because constants are encoded at their stated precision,
round coefficients are cheap, which is the formal version of the "round numbers" hint in TheoPhysEval's inverse problem.

## 3. Four ways a naive inverse reward fails

RLFIND's design is driven by four failure modes, each with a historical precedent.

| Failure mode                                | What the agent learns                                                              | Precedent                                                                                                                                 | Safeguard in RLFIND                                                                                             |
|---------------------------------------------|------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| Reward consistency with the data            | To overfit with any sufficiently flexible family                                   | Ptolemaic epicycles: effectively a Fourier series, which predicted the sky well enough for some 1,400 years                               | Description-length charge; claims judged on conditions chosen by the falsifier, not on the fitted data          |
| Reward agreement with a hidden ground truth | The environment generator's prior instead of nature's                              | Clever Hans, the horse who read his questioners' cues rather than doing arithmetic                                                        | Held-out generator families; adversarial and natural generators; the "Clever Hans gap" as a standing diagnostic |
| Search a fixed hypothesis class             | The best fit inside the wrong class                                                | Le Verrier's Vulcan, proposed to explain the anomalous precession of Mercury's perihelion when the law of gravity itself needed to change | Extensible description language with priced primitives; a rule-bending curriculum                               |
| Score chaotic trajectories pointwise        | Noise: past a few Lyapunov times, the true model scores no better than a wrong one | Lorenz's discovery of sensitive dependence on initial conditions                                                                          | Horizon-aware scoring: trajectories within the predictability horizon, statistics beyond it                     |

The third row deserves emphasis. The same inverse method that let Le Verrier find Neptune from the perturbations of
Uranus produced Vulcan. The method was sound; the hypothesis class was not. In a forward problem, bending the problem
statement is cheating. In an inverse problem it is sometimes the discovery, and no fixed verifier can tell the two
apart. Falsification under an open, priced hypothesis class at least gives the right answer a way to win.

## 4. Environments

### 4.1 Many problems, few verifiers

The scoring core is small, audited, and trusted. It contains the evaluation simulators, the description-language
interpreter and encoder, the falsifier harness, and the statistical tests. Environments, by contrast, are plentiful and
untrusted. Coding agents generate them from parameterized families, and each consists of a hidden system (a simulator, a
condition space, observation channels, and a noise model) behind a query interface.

An environment enters training only after passing validity checks in the core: well-posedness on its condition space,
numerical convergence across two resolutions and two independent solvers, and the existence of a short reference model
that survives the falsifier, which guarantees the environment is fair. A standing red team, itself an agent, is rewarded
for finding exploits in the core. RL is a relentless bug-finder; the design assumes every bug in environment code will
be found, and ensures none of them can become reward.

### 4.2 Counterfactual physics

Since the physics literature is largely memorized, most environment families are counterfactual: laws with perturbed
exponents, nonstandard dissipation operators, altered symmetries, or randomly generated structures constrained to be
physically sensible, meaning dimensionally consistent and with prescribed conservation laws. An agent cannot have
memorized them. TheoPhysEval contains problems of this kind, for example in modified electrodynamics. Rediscovery of
known physics is used only with an explicit contamination analysis.

### 4.3 Adversarial generators

A generator agent is rewarded for producing valid environments on which the current agent fails, subject to the fairness
constraint above. This yields a curriculum that tracks the frontier of the agent's ability, in the spirit of
unsupervised environment design (Dennis et al., 2020), without drifting into unsolvable or merely obscure problems. A
dedicated family of rule-bending generators produces systems whose shortest description lies outside the class the task
description suggests, in the style of TheoPhysEval's unsolved problem.

### 4.4 Curriculum

| Tier | Hidden systems                                                                       | What is inferred                                                                                                                       | Out-of-regime test                                      |
|------|--------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------|
| 0    | Low-dimensional ODEs, maps, delay and implicit equations, rule-bending families      | Governing equations                                                                                                                    | New initial conditions and parameter values             |
| 1    | Partially observed systems with latent variables (cf. Takens, 1981)                  | Hidden state dimension and dynamics                                                                                                    | Prediction of channels revealed only at evaluation      |
| 2    | PDEs: reaction–diffusion, Kuramoto–Sivashinsky, Burgers, 2D turbulence, shell models | PDEs from field data; closures from coarse-grained data                                                                                | Higher Reynolds number, larger domains, longer horizons |
| 3    | Design and control problems                                                          | Conditions producing a target behavior: extreme enstrophy growth, unstable periodic orbits, candidate singularities in model equations | Certification by validated numerics where possible      |
| 4    | Real systems: curated video of physical phenomena, laboratory and observational data | Effective dynamics of real phenomena                                                                                                   | Fresh recordings in new conditions                      |
| 5    | The structural neighborhood of Navier–Stokes                                         | The blowup boundary; candidate funnels                                                                                                 | Computer-assisted proof (Section 6)                     |

Tier 4 is where nature, rather than a coding agent, chooses the generator, which makes it the strongest defense against
learning a generator's prior. Large-scale curation pipelines for physical video, such as NVIDIA's open-source Cosmos
Curator, are a natural source: every clip is a trajectory from dynamics nobody wrote down.

## 5. Training

The base model is a frontier-scale agentic model with tool use. An episode is a research trajectory: explore the system,
run experiments, form and test hypotheses internally, and submit a claim. Because only the reward changes, the standard
RLVR machinery carries over, including group-relative baselines computed over multiple attempts per environment. Tiers
are introduced in order, with replay of earlier tiers.

Simulation dominates cost. We plan to use GPU-native simulators (for example JAX or NVIDIA Warp) and a multi-fidelity
cascade in which cheap low-resolution falsification runs first and expensive high-resolution checks run only on claims
that survive it. The falsifier is trained alongside the agent, but evaluation always uses a stronger, held-out falsifier
portfolio, so the agent cannot overfit to its training adversary.

## 6. Moonshot track: the neighborhood of Navier–Stokes

### 6.1 The neighborhood map

Tao's averaged equation (Tao, 2016) satisfies the energy identity and the standard harmonic-analysis estimates yet blows
up in finite time, so a regularity proof for Navier–Stokes must use finer structure of the actual nonlinearity. Which
structure is load-bearing is an inverse question about a family of equations.

We will build the lattice of model equations that retain different subsets of the features of Navier–Stokes (the energy
identity, vortex-stretching geometry, the Biot–Savart structure, incompressibility, helicity, and the exact trilinear
form), together with continuous dials where they exist, such as the advection parameter of the Okamoto–Sakajo–Wunsch
family (Okamoto et al., 2008) and the dissipation exponent of hypodissipative Navier–Stokes. Agents make claims of the
form "this family blows up from an open set of data in this parameter region, at this rate, with this many unstable
directions," and the falsifier attacks them numerically. The output is the boundary between blowup and regularity, drawn
through the neighborhood of Navier–Stokes.

### 6.2 Funnel search

For singular behavior that survives falsification, agents search for approximate self-similar or discretely self-similar
profiles in suitable, probably anisotropic, similarity variables, and compute their unstable directions. A discretely
self-similar blowup is a periodic orbit of the rescaled equations, and its codimension is its number of unstable Floquet
directions. In these terms, the unforced Navier–Stokes question becomes: does a self-regenerating cascade with finitely
many unstable directions exist?

### 6.3 Certification

Falsification discovers; it does not prove. Surviving candidates are converted into theorems by computer-assisted proof,
combining interval-arithmetic bounds for profile existence and spectral gaps, analytic tail estimates, and a fixed-point
argument. This follows the pipeline of Chen & Hou (2022), with high-precision numerics of the kind developed by Wang et
al. (2025) as the discovery stage. Falsify to discover, certify to prove: every atlas entry is a certified object rather
than a claim to be trusted.

### 6.4 Expectations

Resolving the unforced problem is a long shot. The realistic deliverable is a certified atlas of singularities across
model equations, priced in codimension, with the blowup boundary drawn through the neighborhood of Navier–Stokes, which
tells human analysts which structure a proof must use. Choptuik's critical collapse shows what a complete story can look
like: the numerical discovery of a discretely self-similar critical solution with one unstable mode and universal
scaling (Choptuik, 1993), a renormalization-group explanation, and eventually a computer-assisted existence proof
(Reiterer & Trubowitz, 2019).

## 7. Evaluation

### 7.1 Splits

Environments are held out at three levels: instances from training families, whole families, and whole generators
(environments produced by coding agents, prompts, or human authors never used in training). The drop in performance from
held-out instances to held-out generators, which we call the Clever Hans gap, is the primary diagnostic for whether the
agent has learned physics or its generators' habits.

### 7.2 Benchmarks

| Benchmark                                                                                          | Purpose                                                            | Contamination control                 |
|----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|---------------------------------------|
| Synthetic held-out suites, Tiers 0–3                                                               | Core capability                                                    | Held-out generators                   |
| Expert-authored inverse problems, commissioned from physicists and including rule-bending problems | Human-level difficulty; escape from the suggested hypothesis class | Unpublished; sealed evaluation server |
| Fresh real-world data                                                                              | Transfer to nature                                                 | Recorded after the training cutoff    |
| Rediscovery of known laws                                                                          | Sanity check                                                       | Explicit contamination analysis       |

### 7.3 Metrics

Where a ground truth exists, we measure structural recovery: whether the true model is recovered up to equivalence.
Everywhere, we measure surviving content, description length, query efficiency (surviving content per unit of
experimental cost), calibration (the observed survival rate of claims against the confidence implied by $\kappa$), and
the hypothesis-class escape rate, the fraction of rule-bending problems on which the agent leaves the suggested class
and is right.

### 7.4 Baselines and ablations

Baselines are symbolic regression and sparse identification (Schmidt & Lipson, 2009; Brunton et al., 2016; Udrescu &
Tegmark, 2020; Cranmer, 2023), neural ODE and physics-informed fits followed by symbolic distillation (Cranmer et al.,
2020), frontier agents with tools but no RLFIND training, and agents trained with RLVR on forward problems only.
Ablations remove each safeguard in Section 3 in turn, compare interactive experimentation with fixed datasets at matched
cost, compare weak and strong falsifiers, and sweep $\kappa$.

## 8. Hypotheses

| ID | Hypothesis                                                                                                                                                             | Test                                                                          |
|----|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| H1 | RLFIND-trained agents outperform all baselines on inverse problems from held-out generators.                                                                           | Tier 0–2 held-out-generator suites; structural recovery and surviving content |
| H2 | The description-length charge is necessary for recovering true structure, as opposed to predictive skill alone.                                                        | Ablation; structural recovery at matched predictive skill                     |
| H3 | Interactive experimentation beats passive data at equal cost.                                                                                                          | Matched budgets, active experiments versus fixed datasets                     |
| H4 | An open, priced hypothesis class and a rule-bending curriculum produce a nonzero escape rate on expert-authored rule-bending problems of the kind current agents fail. | Commissioned sealed suite                                                     |
| H5 | Skills learned on counterfactual physics transfer to real systems.                                                                                                     | Tier 4 fresh-data challenge                                                   |
| H6 | The agent produces certified new results on singularity formation in Navier–Stokes model equations.                                                                    | Computer-assisted proofs, independently verified                              |

## 9. Risks and mitigations

| Risk                                             | Mitigation                                                                                                                   |
|--------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| A weak falsifier rewards overclaiming            | Stronger held-out falsifier portfolio at evaluation; calibration audits; $\kappa$ set so that claims require high confidence |
| The falsifier exploits measure-zero pathologies  | Typical-case claims are falsified only by significant violations on sets of positive measure                                 |
| Bugs in the scoring core become reward           | A small audited core; a standing red team; agents submit programs, never predictions                                         |
| The encoding biases which models count as simple | Several encodings, with robustness reported; agent-defined primitives available at a price                                   |
| Simulation cost                                  | GPU-native simulators; multi-fidelity cascades; reduced models in early tiers                                                |
| Contamination                                    | Counterfactual physics; held-out generators; fresh real data                                                                 |
| Proposer–falsifier training is unstable          | Fixed falsifier portfolios early on; population-based training; held-out evaluation adversaries                              |
| Gains on synthetic tiers fail to transfer        | Real-data canaries from Tier 4 introduced early                                                                              |
| The moonshot fails                               | The certified atlas and the neighborhood map are valuable on their own                                                       |

## 10. Milestones and deliverables

| Phase | Months | Milestones                                                                                                                                                            |
|-------|--------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 0     | 0–3    | Scoring core, description language and encoder, Tier 0 generators; baselines; synthetic analogues of TheoPhysEval's inverse failure reproduced against current agents |
| 1     | 3–9    | Tiers 1–2; first RL runs at mid scale; full ablation suite                                                                                                            |
| 2     | 9–18   | Tiers 3–4; frontier-scale training; real-data transfer; first certified atlas entries                                                                                 |
| 3     | 18–36  | Neighborhood map of Navier–Stokes; funnel search; public release of the benchmark and the atlas                                                                       |

The deliverables are RLFIND-Bench, a benchmark of falsifiable inverse problems with a sealed evaluation server; an
open-source scoring core comprising the description language, encoder, falsifier harness, and statistical tests; trained
agents; a certified atlas of singularities in Navier–Stokes model equations; and the accompanying papers.

## 11. Related work

**Equation discovery.** Symbolic regression and sparse identification recover governing equations from data (Schmidt &
Lipson, 2009; Brunton et al., 2016; Udrescu & Tegmark, 2020; Cranmer, 2023), and symbolic distillation can extract laws
from trained networks (Cranmer et al., 2020). These are fixed algorithms over fixed hypothesis classes and passive data,
and they succeed when the right terms are already in the library. RLFIND instead trains a policy that designs
experiments, extends its own hypothesis class, and is judged by survival under attack. The existing methods become tools
for the agent and baselines for the evaluation.

**Reinforcement learning with verifiable rewards.** RLVR (Lambert et al., 2024; DeepSeek-AI, 2025) underlies much of the
recent progress in mathematical and scientific reasoning. RLFIND keeps its machinery, changes the reward, and contains
RLVR as a special case.

**Evaluator-driven discovery.** Search guided by automated evaluators has produced new mathematical constructions at
scale (Georgiev et al., 2025), and transformers trained on synthetic data find global Lyapunov functions where classical
solvers fail (Alfarano et al., 2024). These settings come with a scoring function. RLFIND supplies one where none
exists, by making survival under falsification the score.

**Adversarial verification.** Merlin–Arthur protocols (Babai, 1985) and prover–verifier games (Anil et al., 2021) pair a
powerful untrusted prover with a weak trusted verifier. RLFIND's verifier does not inspect an argument; it runs
experiments. Unsupervised environment design (Dennis et al., 2020) motivates the adversarial generator.

**Scoring and simplicity.** The reward combines minimum description length (Rissanen, 1978) with strictly proper scoring
rules (Gneiting & Raftery, 2007) and operationalizes Popper's (1959) notion of empirical content.

**Singularity formation.** Computer-assisted proof has established finite-time blowup for 3D Euler with a boundary
(Chen & Hou, 2022), and neural-network-assisted numerics have uncovered families of unstable self-similar singularities
(Wang et al., 2025). The moonshot track scales this pipeline and aims it at the neighborhood of Navier–Stokes.

## 12. Open questions

Several questions remain open. We do not yet know whether proposer–falsifier training is stable or cycles the way
adversarial training sometimes does. Content must be defined for heterogeneous observables without letting the choice of
reference forecaster dominate the reward. Description length must be robust enough to the choice of encoding that the
agent's notion of simplicity tracks nature's rather than ours. Domains of validity need a representation expressive
enough to be useful without becoming a new vector for reward hacking. And the deepest question is one TheoPhysEval's
authors raise directly: whether taste, the choice of what is worth modeling, emerges from maximizing surviving content,
or needs a signal of its own.

## References

Alfarano, A., Charton, F., & Hayat, A. (2024). Global Lyapunov functions: a long-standing open problem in mathematics,
with symbolic transformers. *Advances in Neural Information Processing Systems 37*.

Alon, N., Bloom, T. F., Gowers, W. T., Litt, D., Sawin, W., Shankar, A., et al. (2026). Remarks on the disproof of the
unit distance conjecture. arXiv:2605.20695.

Anil, C., Zhang, G., Wu, Y., & Grosse, R. (2021). Learning to give checkable answers with prover–verifier games. arXiv:
2108.12099.

Babai, L. (1985). Trading group theory for randomness. *Proceedings of the 17th Annual ACM Symposium on Theory of
Computing*, 421–429.

Brunton, S. L., Proctor, J. L., & Kutz, J. N. (2016). Discovering governing equations from data by sparse identification
of nonlinear dynamical systems. *Proceedings of the National Academy of Sciences* 113 (15), 3932–3937.

Chandra, A. R., Mahajan, A., Sanyal, A., Wang, M., et al. (2026). TheoPhysEval: Assessing AI agents at autonomous
theoretical physics problem solving. arXiv:2610.16733.

Chen, J., & Hou, T. Y. (2022). Stable nearly self-similar blowup of the 2D Boussinesq and 3D Euler equations with smooth
data. arXiv:2210.07191.

Choptuik, M. W. (1993). Universality and scaling in gravitational collapse of a massless scalar field. *Physical Review
Letters* 70 (1), 9–12.

Cranmer, M. (2023). Interpretable machine learning for science with PySR and SymbolicRegression.jl. arXiv:2305.01582.

Cranmer, M., Sanchez-Gonzalez, A., Battaglia, P., Xu, R., Cranmer, K., Spergel, D., & Ho, S. (2020). Discovering
symbolic models from deep learning with inductive biases. *Advances in Neural Information Processing Systems 33*.

DeepSeek-AI (2025). DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. arXiv:
2501.12948.

Dennis, M., Jaques, N., Vinitsky, E., Bayen, A., Russell, S., Critch, A., & Levine, S. (2020). Emergent complexity and
zero-shot transfer via unsupervised environment design. *Advances in Neural Information Processing Systems 33*.

Fefferman, C. L. (2006). Existence and smoothness of the Navier–Stokes equation. In J. Carlson, A. Jaffe, & A. Wiles
(Eds.), *The Millennium Prize Problems* (pp. 57–67). Clay Mathematics Institute.

Georgiev, B., Gómez-Serrano, J., Tao, T., & Wagner, A. Z. (2025). Mathematical exploration and discovery at scale.
arXiv:2511.02864.

Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the
American Statistical Association* 102 (477), 359–378.

Hadamard, J. (1902). Sur les problèmes aux dérivées partielles et leur signification physique. *Princeton University
Bulletin* 13, 49–52.

Hou, T. Y., & Li, R. (2006). Dynamic depletion of vortex stretching and non-blowup of the 3-D incompressible Euler
equations. *Journal of Nonlinear Science* 16, 639–664.

Lambert, N., et al. (2024). Tülu 3: Pushing frontiers in open language model post-training. arXiv:2411.15124.

Lorenz, E. N. (1963). Deterministic nonperiodic flow. *Journal of the Atmospheric Sciences* 20 (2), 130–141.

Okamoto, H., Sakajo, T., & Wunsch, M. (2008). On a generalization of the Constantin–Lax–Majda equation. *Nonlinearity*
21 (10), 2447–2461.

OpenAI (2026). Finite time blowup for Navier–Stokes. OpenAI preprint, September 2026.

Popper, K. R. (1959). *The Logic of Scientific Discovery*. London: Hutchinson.

Reiterer, M., & Trubowitz, E. (2019). Choptuik's critical spacetime exists. *Communications in Mathematical Physics*
368, 143–186.

Rissanen, J. (1978). Modeling by shortest data description. *Automatica* 14 (5), 465–471.

Schmidt, M., & Lipson, H. (2009). Distilling free-form natural laws from experimental data. *Science* 324 (5923), 81–85.

Takens, F. (1981). Detecting strange attractors in turbulence. In *Dynamical Systems and Turbulence, Warwick 1980*,
Lecture Notes in Mathematics 898, 366–381. Springer.

Tao, T. (2016). Finite time blowup for an averaged three-dimensional Navier–Stokes equation. *Journal of the American
Mathematical Society* 29 (3), 601–674.

Udrescu, S.-M., & Tegmark, M. (2020). AI Feynman: A physics-inspired method for symbolic regression. *Science Advances*
6 (16), eaay2631.

Wang, Y., et al. (2025). Discovery of unstable singularities. arXiv preprint.
