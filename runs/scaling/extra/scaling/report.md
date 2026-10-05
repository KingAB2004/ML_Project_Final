# Scaling study: Qwen2.5 0.5B / 3B / 7B, same QLoRA recipe, 1000-profile corpus

Test metrics: every size plays against the same Qwen2.5-7B seeker (scripts/scaling_eval.sh), scored by the Mistral-Nemo judge. Validation losses of the two arms are not comparable to each other (the with_thoughts target also holds the thoughts).

| size | arm | N (non-emb) | best val loss | best epoch | success | ip | pri | PRI (counterfactual) |
|---|---|---|---|---|---|---|---|---|
| 0.5B | with_thoughts | 0.36B | 0.8420 | 2 | - | - | - | - |
| 0.5B | wo_thoughts | 0.36B | 0.8755 | 2 | - | - | - | - |
| 3B | with_thoughts | 2.77B | 0.6872 | 2 | - | - | - | - |
| 3B | wo_thoughts | 2.77B | - | - | - | - | - | - |
| 7B | with_thoughts | 6.53B | 0.6117 | 2 | - | - | - | - |
| 7B | wo_thoughts | 6.53B | - | - | - | - | - | - |

- with_thoughts: best val loss ~ 7.14 N^-0.108 (1 residual df)

![scaling_loss.png](scaling_loss.png)
![training_curves.png](training_curves.png)
![training_curves_0.5B.png](training_curves_0.5B.png)
![training_curves_3B.png](training_curves_3B.png)
![training_curves_7B.png](training_curves_7B.png)

Every number is measured on simulated seekers and read by an LLM judge.
