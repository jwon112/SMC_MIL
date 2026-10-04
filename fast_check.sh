#!/usr/bin/env bash
# Small validation only. No full extraction is launched or stopped.
set -e
cd /home/jupyter/image_team/projects/SMC_MIL
echo '--- CURRENT HOST / EXTRACTION PROCESSES ---'
hostname
ps -eww -o pid,ppid,stat,args | grep -E '[r]un_pathomics_parallel.py|[r]un_wsi_pathomics.py|[r]un_pathomics_fast.py run' || true
echo '--- VERSION / REUSE CHECK ---'
/home/jupyter/image_team/miniconda3/envs/clam_latest/bin/python -u run_pathomics_fast.py check --workers 30 --allow-verified-extent-mpp
echo '--- SMALL REAL-IMAGE PARITY AND TIMING TEST ---'
/home/jupyter/image_team/miniconda3/envs/clam_latest/bin/python -u run_pathomics_fast.py verify --patches 8 --allow-verified-extent-mpp
echo 'Check finished. No full extraction launched. Review timing before running fast_run.sh.'
