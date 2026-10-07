# Machine-learning concepts used in this project

Every concept below is tied to the file that implements it and to what has been run with it so far (status as
of 7 Oct 2026). "Ran" means results exist under `results/` or are in a README; "implemented" means the code and
its test exist but no result has been produced yet.

Contents
1. Large language models and generation
2. Synthetic data generation and user simulation
3. Data quality and diversity measures
4. Supervised fine-tuning (SFT) and rationale supervision
5. Parameter-efficient fine-tuning: LoRA and QLoRA
6. Optimisation and training mechanics
7. Scaling laws
8. Information theory: pointwise V-information (PVI)
9. Linear algebra of the LoRA updates: SVD, rank, subspaces
10. Multi-agent decomposition and self-refinement
11. Grounding and the disclosure ladder
12. Conformal risk control (the gate)
13. Need-state memory: belief state, decay, pruning, retrieval
14. Memory baselines: summary, dense retrieval, event memory
15. Evaluation: LLM-as-judge and psychometric scales
16. Counterfactual attribution (PRI)
17. Statistics: bootstrap, paired designs, ablations, held-out splits
18. Further analyses (implemented, not yet run): conformal prediction sets, temperature scaling, expected
    information gain, Markov chains, survival analysis, psychometrics, calibration

---

## 1. Large language models and generation

**Decoder-only transformers, instruction-tuned.** Every model is an autoregressive transformer that predicts the
next token given all previous ones, p(x_t | x_<t). Instruction-tuned variants follow a chat format of system,
user and assistant messages.
- Models: Qwen2.5-0.5B / 3B / 7B-Instruct and Qwen3-4B (supporter, seeker simulator, agents), Mistral-Nemo-12B
  (profile writer and judge, a different model family on purpose, see section 15).
- Chat templates: `tokenizer.apply_chat_template(...)`; for Qwen3 `enable_thinking=False` so the model does not
  emit its own hidden reasoning block (`src/train.py`, `extra/_lm.py`, `src/llm.py`).

**Sampling.** Generation draws tokens with temperature and nucleus (top-p) sampling, e.g. temperature 0.85,
top-p 0.95 for corpus dialogue; scoring uses greedy or low-temperature decoding. Every record is seeded by its
id (`common.rng_for`, `derive_seed`) so a rerun reproduces it.

**Structured output.** Agents and the annotator must return JSON (needs, spans, statuses); `llm.structured`
parses it and marks `parse_failed` instead of crashing, and model JSON is treated as untrusted input.

**Serving.** `src/llm.py` is the only module that loads weights: Ollama (quantised GGUF, the lab), Hugging Face
transformers (4-bit base + LoRA adapter, switched on only for the supporter via `use_adapter`), and an echo
backend for tests. Responses are cached by a hash of the prompt and sampling settings.

Status: used in every stage.

## 2. Synthetic data generation and user simulation

**Seeded generation.** Real situations (EmpatheticDialogues) seed every profile, and the ESConv problem-type
taxonomy organises them (`src/seeds.py`). Seeding keeps synthetic data anchored to real distributions.

**Latent-state user simulation.** Each profile carries hidden state the supporter never sees: a 3-level need
chain (surface feeling, intermediate need, terminal need), a persona, disclosure triggers and blockers, and a
resistance level (`src/profiles.py`). The simulated seeker (`src/dialogue.py`, prompts `user_proactive.md`,
`user_reactive.md`, `user_corrective.md`) conditions on that state, so the "ground truth" of every dialogue is
known, which makes metrics like Success computable. Three simulator policies: proactive (opens up when handled
well), reactive (only reacts), corrective (openly rejects a wrong guess).

**Multi-session processes.** 2-4 sessions per person, gaps sampled log-uniformly over 1-56 days, and a
stochastic state transition between sessions (resolved 35 %, intensified 45 %, displaced 20 %) that rewrites
the need chain (`src/sessions.py`). This is a simple discrete-state process over the person's need.

Status: ran - 1,000 profiles (720 train / 100 val / 100 test / 80 calibration), 2,369 sessions (762 intensified,
508 resolved, 298 displaced follow-ups).

## 3. Data quality and diversity measures

**Rule- and model-based filtering** (`src/filter.py`): structure checks, leakage of the hidden need into the
seeker's early turns, crisis content, active-listening quality. Kept 2,369 sessions.

**Diversity metrics** (`extra/data_quality.py`, the paper's Table 8):
- *Self-BLEU-n*: BLEU of each description against the others; high = repetitive. Ours 0.702 (BLEU-2), 0.398
  (BLEU-4) vs ExTES 0.724 / 0.450.
- *Distinct-n*: unique n-grams / total n-grams. Ours 0.256 vs ExTES 0.237.
- *Shannon entropy* of the word distribution, H = -sum p(w) log2 p(w). Ours 8.38 bits vs ExTES 7.80.

**Label-collapse check** (`src/annotate.py`): strategy labels clustered by Jaccard similarity; if the top 3
clusters cover > 80 % of labels the supervision signal has collapsed.

Status: filtering and Table 8 ran; dialogue-quality judge scores (Table 2) queued.

## 4. Supervised fine-tuning (SFT) and rationale supervision

**Causal language-model loss with target masking.** Each example is (input, target). Loss is next-token
cross-entropy on the target tokens only; input tokens get label -100 (`src/train.py: encode_example`).
Examples longer than the 2,048-token window are dropped and counted.

**Rationale supervision (learning to think before replying).** Two arms trained on identical replies:
- *with thoughts*: target = `<analysis>...</analysis><strategy>...</strategy><response>...</response>`
- *without thoughts*: target = `<response>...</response>`
The thoughts are learned, not prompted - a form of chain-of-thought / rationale distillation from annotations
(`src/build_sft.py`, `src/annotate.py`).

**Leakage-free splits.** Splits are by person, not by turn, so no test person's words are ever in training
(`build_sft.split_assignment`). Test profiles never get corpus dialogues at all.

**Memory-aware SFT, Option B** (`src/futurememory.py`): replay the corpus through the need-state memory, add
the brief to every input, re-annotate sessions 2+ with the brief visible. Implemented and tested; not run.

Status: ran - 18,821 train / 2,608 val examples per arm.

## 5. Parameter-efficient fine-tuning: LoRA and QLoRA

**LoRA.** The pretrained weight W is frozen; each adapted matrix learns a low-rank update
dW = (alpha / r) * B A, with B (d_out x r), A (r x d_in), r << d. Here r = 32, alpha = 64, dropout 0.05, on all
attention and MLP projections (q, k, v, o, gate, up, down). Only a few tens of millions of parameters train.

**QLoRA.** The frozen base is stored in 4-bit NF4 (NormalFloat, quantisation levels matched to a normal
distribution of weights) with double quantisation of the scales; computation happens in bf16/fp16. This is what
lets a 7B model train on a 12 GB RTX 3060 and a 3B on an 8 GB laptop GPU.

**Adapter switching at inference.** One resident 7B serves every role; the adapter is enabled only for the
supporter's calls.

Status: ran - Qwen2.5 0.5B / 3B (laptop RTX 4060), 7B (lab RTX 3060), both arms; Qwen3-4B training.

## 6. Optimisation and training mechanics

- AdamW in paged 8-bit form (`paged_adamw_8bit`): optimizer state quantised and paged to CPU memory under
  pressure.
- Learning rate 2e-4, 3 % linear warm-up, cosine decay.
- Micro-batch 1 with 16-step gradient accumulation (effective batch 16).
- Gradient checkpointing: activations recomputed in the backward pass to save memory.
- Mixed precision (bf16 where supported, else fp16).
- Shuffled mini-batches: the trainer reshuffles all turn-level examples each epoch, mixing people and sessions.
- Model selection / early stopping: validation loss each epoch, best checkpoint kept. Epoch 2 was best for all
  six Qwen2.5 runs; epoch 3 overfits (validation loss rises while training loss keeps falling).

## 7. Scaling laws

Validation loss vs non-embedding parameter count N fitted as a power law L(N) = a N^-b by linear regression in
log-log space (`extra/scaling.py`). With thoughts: L = 7.14 N^-0.108 over 0.5B / 3B / 7B; on the identical reply
tokens the exponents are 0.167 (with thoughts) and 0.187 (without), with bootstrap CIs.

Status: ran.

## 8. Information theory: pointwise V-information (PVI)

Usable information (Ethayarajh, Choi & Swayamdipta 2022) measures how much an input X tells a *given model
family* about Y. Per example:

  PVI_i = log2 p_with(y_i | x_i, thoughts_i) - log2 p_wo(y_i | x_i)

with the two fine-tuned adapters as the two models, both teacher-forced on the identical reply tokens
(`extra/pvi.py`, `extra/_lm.py`). The mean over examples is the V-information in bits. Related quantities:
per-token negative log-likelihood (NLL), and the share of examples with PVI < 0 (annotations that point away
from the reply, i.e. candidates for label noise).

Results: 8.91 / 7.36 / 3.31 bits per turn (0.5B / 3B / 7B); deeper ladder turns gain more; split by session the
thoughts add 3.0-3.7 bits at 7B in every session (`extra/pvi_by_session.py`). The 7B drop is explained by the
corpus replies being written by Qwen2.5-7B itself (self-generated-data effect).

## 9. Linear algebra of the LoRA updates

`extra/lora_geometry.py`:
- **SVD without forming dW**: QR of B and of A^T, then the SVD of the small r x r core gives dW's singular values.
- **Effective rank**: exp of the entropy of the normalised singular values (how many directions are used).
- **Stable rank**: ||dW||_F^2 / ||dW||_2^2 (how many directions dominate).
- **Eckart-Young truncation**: replace every dW by its best rank-k approximation and re-measure validation loss;
  the k needed for 95 % of the gain falls with size (16, 4-8, 2-4).
- **Subspace overlap** between the two arms' top-k directions (principal angles), compared against random
  subspaces of the same size.

Status: ran for all three sizes.

## 10. Multi-agent decomposition and self-refinement

The decomposed supporter (`src/agents.py`) splits one reply into roles, each a separate LLM call with its own
prompt:
- Analyzer: emotional state and candidate needs, each with quoted evidence, depth and parent.
- Strategist: the move and its depth.
- Responder: the draft.
- Critic: grounding checks and a predicted intrusiveness score.
A *self-refinement loop* sends a rejected draft back with feedback (up to 2 revisions), then falls back to a safe
reflective reply. The monolithic baseline (`src/baselines/listener_mono.py`) does all of it in one call, so the
2 x 2 design (architecture x gate) isolates the value of decomposition.

Status: ran in cells A-D (C/D in progress), calibration, memory arms.

## 11. Grounding and the disclosure ladder

**Evidence grounding** (`src/grounding.py`): every claim about the seeker must cite a verbatim span of their own
words (turn index + quote); unsupported inferences, denied inferences and stale references are typed
violations. Comparable to attribution / citation constraints for faithful generation.

**Disclosure ladder** (`src/pacing.py`): L0 reflect, L1 explore, L2 tentative guess, L3 name the need. The rung is
capped by the Analyzer's confidence, and L3 additionally requires a confirmed need in memory - a hand-designed
policy constraint on the action space.

## 12. Conformal risk control (the gate)

Distribution-free control of the rate of intrusive replies (`src/conformal.py`, `src/calibrate.py`):
- Nonconformity score per draft: s = 0.7 x Critic-predicted IP + 0.3 x grounding-violation mass.
- A calibration set of rollouts on held-out calibration profiles gives the empirical risk R(lambda) of releasing
  drafts with s <= lambda (risk = share of released turns with IP > tau = 0.20).
- Upper confidence bounds on the risk: Hoeffding and Bentkus (the "HB" bound), at confidence 1 - delta, delta = 0.1.
- lambda-hat = the largest threshold whose upper bound stays <= alpha. Guarantee: with probability >= 1 - delta
  the intrusive rate among released turns is <= alpha, without assumptions on the model.

Results: alpha 0.05 / 0.10 / 0.20 give lambda-hat 0 / 0.2 / 1.0 and release 13 / 73 / 100 %; with the gate
(alpha 0.10) intrusive turns fell from 4.0 % to 2.3 %.

## 13. Need-state memory: belief state, decay, pruning, retrieval

`src/memory.py`:
- **Belief state as a graph of hypotheses.** A forest: each session's surface feeling at depth 0, needs at depth
  1-3 linked to the shallower need they explain. Siblings are competing explanations (rival hypotheses).
- **Evidence-based updates.** Propose only with a quoted span; confirm only after the person agrees in their
  latest turn (then rivals x 0.7); a denial is stored permanently with the quote, blocks re-proposal, and makes
  the subtree dormant; resolve and reinstate follow change over time.
- **Exponential forgetting with reinforcement** (spaced-repetition style): confidence x 0.5^(gap / half-life);
  half-life 14 days, x 3 once confirmed, x 1.5 for each later session that brings the need up again, capped at
  180 days.
- **Pruning**: below 0.15 a node leaves the brief; below 0.05 it leaves the graph, deepest first; a strong child
  is re-attached to its grandparent.
- **Bounded retrieval**: only a brief of at most 320 tokens (best chain, rivals, untested needs, denials) enters
  the prompt - retrieval as "what we do not know yet", not a transcript.
- **Lexical similarity** for matching needs: Jaccard over content words with framing words removed and a
  7-letter prefix stem.
- **Event-sourced log**: state is a fold over an append-only transition log, so every belief change is
  auditable.

Status: flat version ran (`mem_needstate`, Success 0.381); tree + fade / prune ran in the lab trial (true need
confirmed in session 3) and in the fine-tuned 7B memory run (running); v2 arms queued.

## 14. Memory baselines

`src/baselines/`:
- `memory_none`: no cross-session state (the control).
- `memory_summary`: a running prose summary regenerated each session (structure vs prose).
- `memory_dense`: dense retrieval - past turns embedded (sentence-transformers, or a bag-of-words cosine
  fallback), top-k by cosine similarity to the latest turn (the standard RAG pattern).
- `memory_event`: event memory with recency decay, retrieved by the question.

Status: implemented; queued as the v2 memory arms and in ES-MemEval (`extra/memeval_qa.py`).

## 15. Evaluation: LLM-as-judge and psychometric scales

**LLM-as-judge** (`src/judge.py`, `src/metrics.py`): a model rates each dialogue or turn on fixed item lists.
- *Judge separation*: the judge (Mistral-Nemo) must be a different family from the generator (Qwen), enforced in
  code (`assert_judge_separate`), to limit self-preference bias.
- *Hidden-information redaction*: the judge sees only what each scale needs (e.g. the terminal need for Success).
- Likert items (1-7) parsed, averaged and rescaled to 0-1 or 0-100.

**Scales**:
- Success: did the supporter put the hidden need into words (1-7).
- AELS: the Active-Empathic Listening Scale (sensing, processing, responding), item wording verbatim.
- CRS (Aff, Neg) and RAC (Sup, Man): the baseline paper's affective / negative reaction and supportiveness /
  management dimensions.
- Basic: fluency, diversity, empathy, information, human-likeness, skill.
- IP (intrusiveness) and PRI (reactance), turn-level.

Status: ran for every finished arm.

## 16. Counterfactual attribution (PRI)

`src/counterfactual.py`: to attribute the seeker's resistance to the supporter's move rather than to the
situation, each sampled turn is replayed twice from the same random seed - once with the actual move, once with
a length-matched neutral reflection - and the seeker's reaction is judged in both. PRI = mean(factual) -
mean(control), with a paired bootstrap CI over the seed-matched pairs. Positive PRI = the move provoked
resistance beyond what a neutral reply would have.

Status: ran for the test runs and baselines (e.g. 7B with thoughts 0.018 [0.008, 0.029]).

## 17. Statistics and experimental design

- **Bootstrap confidence intervals** (10,000 resamples), resampling profiles (cluster bootstrap) because
  sessions and turns of one person are correlated.
- **Paired / seed-matched comparisons** (PRI; profile-by-profile comparisons between arms).
- **Ablations and factorial design**: one arm = one set of component switches (`configs/arms.yaml`); the 2 x 2
  crosses architecture (monolithic vs decomposed) with the gate (off vs on); memory arms swap only the memory;
  with vs without thoughts swaps only the target.
- **Held-out evaluation**: 100 test people never seen in training; 80 calibration people used only for the gate.
- **Teacher forcing vs free generation**: PVI and validation loss score known replies; Success and IP come from
  live rollouts, and both views are reported.

## 18. Further analyses (implemented and tested, not yet run on the 1,000-person data)

- **Conformal prediction sets with temperature scaling** (`extra/need_sets.py`): the Analyzer scores K candidate
  needs, scores are calibrated by a fitted temperature (softmax(beta r)), and split-conformal sets guarantee
  the true need is in the set with probability >= 1 - alpha.
- **Expected information gain** (`extra/eig_probe.py`): choose the next move by the expected reduction in
  entropy of the belief over needs, minus a reactance penalty - an active-learning view of questioning.
- **Absorbing Markov chain** (`extra/markov.py`): seeker states (guarded, opening, withdrawn, disclosed) per turn;
  transition matrix, fundamental matrix, expected turns to disclosure.
- **Survival analysis** (`extra/survival.py`): time to first disclosure with right-censoring; Kaplan-Meier curves,
  log-rank test, Cox proportional-hazards model.
- **Psychometrics** (`extra/psychometrics.py`): reliability, Horn's parallel analysis, maximum-likelihood factor
  analysis fitted by EM with BIC, and item response theory, to check that IP and PRI measure what they claim.
- **Calibration of memory beliefs** (`extra/memory_calibration.py`, queued): Brier score with Murphy's
  decomposition (reliability, resolution, uncertainty), log loss, expected calibration error over 10 bins.
