temp_results: every comparison so far, one folder per question (rebuilt by extra/build_temp_results.py)

01_dataset_quality__ours_vs_ExTES_vs_ESConv/   ours vs ExTES vs ESConv
    dataset_stats.txt
    table2_dialogue_quality.txt
    table8_profile_diversity.txt
02_finetuning__validation_loss_by_model_size/   validation loss by model size
    report.txt
    scaling_loss.png
    training_curves.png
    training_curves_0.5B.png
    training_curves_3B.png
    training_curves_7B.png
    validation_loss_by_epoch.txt
03_information_in_thoughts__PVI/   PVI
    pvi_by_session.png
    pvi_by_session.txt
    pvi_by_size.png
    pvi_by_size.txt
    pvi_hist.png
    report.txt
04_lora_geometry__how_the_update_uses_its_rank/   how the update uses its rank
    effective_rank.png
    geometry.txt
    norm_and_similarity_by_depth.png
    rank_truncation.png
    rank_truncation.txt
    report.txt
05_single_session_test__finetuned_vs_untuned/   finetuned vs untuned
    scaling_test.png
    scores.txt
    thoughts_gain.png
06_architecture_and_gate__2x2_cells/   2x2 cells
    cells.txt
    dialogue_diagnostics.png
    gate_calibration.png
    gate_calibration.txt
    ip_per_turn.png
    pri_counterfactual.png
    score_distributions.png
    scores_by_arm.png
07_multi_session__finetuned_7B_with_and_without_memory/   finetuned 7B with and without memory
    overall.txt
    success_by_change.txt
    success_by_gap.txt
    success_by_session.png
    success_by_session.txt
08_need_state_memory__untuned_flat_run/   untuned flat run
    memory_eval_summary.txt
    report.txt
    success_by_gap.png
09_one_person_ablation__p000394/   p000394
    ablation.txt
10_v2_reference__50_profile_run/   50 profile run
    v2_baselines_2x2_and_finetune.txt
