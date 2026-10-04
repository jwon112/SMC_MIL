#!/usr/bin/env bash
# Starts only the requested all-WSI fast extraction; never kills other jobs.
set -e
cd /home/jupyter/image_team/projects/SMC_MIL
if ps -eww -o pid,ppid,args | grep -E '[r]un_pathomics_parallel.py|[r]un_wsi_pathomics.py|[r]un_pathomics_fast.py run'; then
    echo 'STOP: an extraction process is already visible. Inspect it before starting another.'
    exit 1
fi
/home/jupyter/image_team/miniconda3/envs/clam_latest/bin/python -u run_pathomics_fast.py check --workers 30 --allow-verified-extent-mpp --require-validation
nohup /home/jupyter/image_team/miniconda3/envs/clam_latest/bin/python -u run_pathomics_fast.py run --workers 30 --allow-verified-extent-mpp >> pathomics_all_wsi_fast.log 2>&1 &
echo "Fast extraction launch PID: $!"
echo 'Log command: tail -f pathomics_all_wsi_fast.log'
echo 'A matching successful fast_validation.json is required; otherwise the launcher stops.'
