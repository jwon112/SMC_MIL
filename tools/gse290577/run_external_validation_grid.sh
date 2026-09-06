#!/usr/bin/env bash
set -u

WORK_ROOT=${WORK_ROOT:-/home/jupyter/data/image_team/GSE290577_work}
FEATURE_ROOT=${FEATURE_ROOT:-/home/jupyter/image_team/projects/SMC_MIL/data/features/uni_v2/GSE290577}
OUTPUT_ROOT=${OUTPUT_ROOT:-results/gse290577_external/weakunique3}
DEVICE=${DEVICE:-cuda}

tasks=(acr_any acr_high amr_positive any_rejection)
prefixes=(acr_0r_vs_rest acr_high_grade amr_positive any_rejection)
tags=(0p25 0p5 1 2)
checkpoint_mpps=(0p25 0p50 1p00 2p00)
levels=(l0 l1 l2 l3)
magnifications=(40x 20x 10x 5x)

mkdir -p "$OUTPUT_ROOT"
failures=0

for i in "${!tasks[@]}"; do
    task=${tasks[$i]}
    prefix=${prefixes[$i]}

    for j in "${!tags[@]}"; do
        tag=${tags[$j]}
        checkpoint_dir="results/smc_${prefix}_${levels[$j]}_${checkpoint_mpps[$j]}mpp_${magnifications[$j]}_uni2_clamsb_cv3val_weakunique3_s1"
        output_dir="$OUTPUT_ROOT/${task}_${tag}mpp"

        echo "[RUN] task=$task resolution=${tag}mpp"
        cohorts=(--cohort wsi_he "$WORK_ROOT/manifests/gse290577_wsi_he.csv" "$FEATURE_ROOT/${tag}mpp/wsi")
        if [[ $tag == 0p25 ]]; then
            cohorts+=(--cohort core "$WORK_ROOT/manifests/gse290577_core_inventory.csv" "$FEATURE_ROOT/${tag}mpp/core")
        fi

        if python evaluate_external_clam.py \
            --checkpoint-dir "$checkpoint_dir" \
            --task "$task" \
            "${cohorts[@]}" \
            --output-dir "$output_dir" \
            --embed-dim 1536 \
            --threshold 0.5 \
            --bootstrap 2000 \
            --device "$DEVICE"; then
            echo "[DONE] task=$task resolution=${tag}mpp"
        else
            status=$?
            echo "[FAIL] task=$task resolution=${tag}mpp exit_code=$status" >&2
            failures=$((failures + 1))
        fi
    done
done

echo "Completed grid with $failures failed evaluation(s)"
(( failures == 0 ))
