# Summaries: one folder per result

Short written results (tables, text, plots), copied from `temp_results/` by `scripts/collect_results.sh`. The SOP map
in `../README.md` ("Where each SOP item's results are") says which SOP section each one answers and gives its headline.

| Folder | SOP part | What |
|---|---|---|
| `01_dataset_quality__ours_vs_ExTES_vs_ESConv` | Enhancement 1, Section IV | corpus statistics and quality vs ExTES / ESConv |
| `02_finetuning__validation_loss_by_model_size` | Section III-E | QLoRA fine-tuning, losses by size, scaling fits |
| `03_information_in_thoughts__PVI` | Expected outcome 1 | usable information in the annotated thoughts |
| `04_lora_geometry__how_the_update_uses_its_rank` | Expected outcome 1 | how the LoRA update uses its rank |
| `05_single_session_test__finetuned_vs_untuned` | Expected outcome 1 | fine-tuned vs untuned on the test people |
| `06_architecture_and_gate__2x2_cells` | Enhancement 2, Expected outcome 2 | the 2 x 2: decomposition x gate |
| `07_multi_session__finetuned_7B_with_and_without_memory` | Expected outcome 4 | fine-tuned 7B with / without memory |
| `08_need_state_memory__untuned_flat_run` | Enhancement 4 | need-state memory, first run |
| `09_one_person_ablation__p000394` | Expected outcome 1 | one person through every arm |
| `10_v2_reference__50_profile_run` | reference | the 50-profile v2 run |
| `11_multi_session__all_7B_systems` | Expected outcomes 2, 4 | every 7B system side by side |
| `12_architecture_diagrams` | Enhancement 2 | pipeline figures |
| `13_memory_comparison__untuned_4_arms` | Enhancement 4, Expected outcome 4 | memory types compared, ES-MemEval |
| `14_gate_recovery_vs_cost` | Expected outcome 3 | the critic's recovery-vs-cost curve |
| `15_activation_steering` | added | steering vs its control, cells A-D, step 12 |
| `16_corpus_comparison__ours_vs_ExTES` | Section IV | ExTES-trained vs ours-trained, both people sets |
| `17_gate_violation_rate` | Enhancement 2, scope of the guarantee | violation rate in and out of distribution |
| `18_preference_alignment_dpo` | added | DPO vs SFT (when the run lands) |
| `19_llm_psychology` | added | sycophancy, EmoBench, dependence |
| `20_step12_finetuned_generator` | Enhancement 2 | fine-tuned Generator in the gated pipeline |
| `21_judge_judge_agreement` | Expected outcome 3 (added check) | second judge vs main judge (when the run lands) |
| `22_qualitative_examples` | Enhancements 2-4 | rule-selected examples |
| `23_multiple_comparisons` | all | Holm-corrected headline contrasts |
| `24_cost_per_turn` | Expected outcome 3 (added) | model calls per turn vs IP bought |
| `25_judge_machine_check` | Expected outcome 3 (added check) | the same judge on two machines: how far the scores move |
