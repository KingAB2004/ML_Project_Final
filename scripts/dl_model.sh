#!/bin/bash
# Download the base model for fine-tuning, retrying through the flaky connection to the Hub.
cd ~/COCCON_NEW && source coccon/bin/activate
export HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_ETAG_TIMEOUT=60
for i in $(seq 1 60); do
  echo "attempt $i $(date)"
  hf download Qwen/Qwen2.5-7B-Instruct && { echo DL_OK; exit 0; }
  sleep 30
done
echo DL_FAILED; exit 1
