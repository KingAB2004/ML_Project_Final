# Scaling study: Qwen2.5 0.5B / 3B / 7B, same QLoRA recipe, 1000-profile corpus

Test metrics: every size plays against the same Qwen2.5-7B seeker (scripts/scaling_eval.sh), scored by the Mistral-Nemo judge. Validation losses of the two arms are not comparable to each other (the with_thoughts target also holds the thoughts).

| size | arm | N (non-emb) | best val loss | best epoch | success | ip | pri | PRI (counterfactual) |
|---|---|---|---|---|---|---|---|---|
| 0.5B | with_thoughts | 0.36B | 0.8420 | 2 | - | - | - | - |
| 0.5B | wo_thoughts | 0.36B | 0.8755 | 2 | - | - | - | - |


![scaling_loss.png](scaling_loss.png)

Every number is measured on simulated seekers and read by an LLM judge.
