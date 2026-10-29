# Substitutions

Every deviation from the configuration planned in the SOP (Section IV-B Resources, the Appendix), with its reason and,
where it could move a result, how large that effect was measured to be.

| date | component | planned (SOP) | actual | reason / measured effect |
|---|---|---|---|---|
| Oct 2-21 | hardware | a single 12 GB GPU (RTX 3060) | lab RTX 3060 12 GB (10.50.28.201); laptop RTX 4060 8 GB (0.5B / 3B fine-tunes, PVI, smoke tests); 10.10.10.104 RTX 3060 12 GB; 10.10.2.41 RTX 3060 12 GB until 16 Oct | the lab was down 9-16 Oct (an interrupted NVIDIA driver upgrade); work was spread over the other machines. Every run still fits one 12 GB card, one model resident at a time, as planned |
| all | generation and role-play model | one instruction-tuned open model for corpus generation and user role-play | Qwen2.5-7B-Instruct for both, and for the four agent roles (switched by role prompt) | as planned |
| all | judge | a larger open model from a different family, distinct from supporter and simulator | Mistral-Nemo-Instruct-2407 (12B, Q4_K_M via Ollama) | as planned; `src/judge.py` refuses a judge that shares the supporter's or simulator's model |
| Oct 20-21 | second judge | - | Llama-3.1-8B-Instruct (Q4_K_M) rescoring 9 key arms | added check of the judge choice (`summaries/21_*`); it does not replace judge-human agreement |
| all | inference server | a batched inference server | Ollama (GGUF Q4_K_M) for un-tuned generation and all judging; HF transformers with bitsandbytes nf4 wherever a LoRA adapter or an activation hook is needed (step 12, steering, DPO, the fine-tuned single-session tests) | Ollama cannot load our adapters or hook layers. The vLLM backend in `src/llm.py` was not used for reported runs. **Measured effect:** the backend alone raises the untuned one-call listener's IP by +0.017 [0.009, 0.025] (transformers vs Ollama; survives Holm), so cross-backend comparisons (step 12 vs cell D) carry this confound |
| Oct 18-23 | judge serving | one judge | the same Mistral weights (digest daf673741712) served by Ollama 0.35 on the lab and 0.18 on 10.10.10.104 | **measured effect:** session-level dependence and sycophancy scales shift a lot between the two (e.g. reliance 0.08 vs 0.27 on the same sessions); the psychology probes were therefore all re-judged on 10.10.10.104. The turn-level IP check is in `summaries/25_judge_machine_check` |
| all | conformal gate | a conformal prediction library | own implementation (`src/conformal.py`: risk control with Hoeffding / Bentkus upper bounds, alpha 0.10, tau 0.20, delta 0.10) | small and auditable; the SOP's guarantee (a bound on released-and-intrusive turns) is what it implements |
| Oct 4-6 | fine-tuning | an open base model fine-tuned with QLoRA | Qwen2.5 0.5B / 3B / 7B, LoRA r 32, alpha 64, 3 epochs, lr 2e-4, seq 2048, with and without the thoughts | the 7B is the planned supporter; 0.5B and 3B added for the scaling study (and a Qwen3-4B control in the PVI analysis) |
| Oct 7 | corpus scale | "final scale is expected to follow the yield" | 1,000 profiles, 2,369 sessions (2-4 per profile, 8-12 turns), 21,429 annotated supporter turns | yield of the pipeline |
| Oct 7 | profile seeding | EmpatheticDialogues, LLM-synthesized as fallback | EmpatheticDialogues (85.5 % of screened seeds passed the need-chain screen) | the planned route worked; no fallback |
| Oct 16 | memory baselines | session-isolated, flat summary, dense retrieval, event-level [9] | the first three; the event-level arm was dropped | team decision; the remaining three cover isolation, summarization and retrieval |
| Oct 16 | memory evaluation sets | our sequences, ExTES user profiles, ES-MemEval | our sequences and ES-MemEval | the ExTES-profile memory runs (~2 days of GPU) were not run by decision; ES-MemEval already tests memory on data we did not generate |
| Oct 16-17 | ExTES as training corpus | ExTES as a reactive baseline and second evaluation set | 7B trained on ExTES sampled to 15,000 examples (0.7 x our corpus, whole dialogues), from the untuned base | matches training-set size so the comparison is about the data, not its size |
| Oct 16-17 | step 12 gate calibration | calibrate on a held-out split from the same generator | calibrated on 10.10.2.41 (transformers 4.57), deployed on the lab (transformers 5.18) | the lab replaced the lost machine; the realised violation rate (0.091 [0.074, 0.112] vs alpha 0.10) is reported as measured |
| Oct 18-20 | ESC-RANK | the paper's own scorer | ESC-RANK on InternLM2-chat-7B in its own venv (transformers 4.41, sentencepiece 0.2.0, protobuf) on 10.10.10.104; Inf answers re-asked with a 16-token budget | ESC-RANK's code needs an older stack than the project venv; the 8-token budget cut every Inf answer |
| all | arm pairing | - | most arms seed sessions by an id that includes the arm name, so arms are compared unpaired; steering vs its control and DPO vs SFT share seeds and are paired | unpaired comparisons are conservative (wider CIs) |
| - | judge-human agreement | a stratified subsample rated by the team | not done | needs human raters; the sampler, forms and agreement script are ready in `human_eval/` |

Additions beyond the SOP (activation steering, DPO, LLM psychology probes, second judge, multiple-comparison
correction, cost per turn, qualitative examples) are listed in `results/EXTRA_STUDIES.txt`.
