#!/usr/bin/env bash
set -euo pipefail
CFG="/home/x-cli32/chaohan/projects/open-r1/src/open_r1/sftSPLLM.yaml"
echo "Using config: $CFG"

accelerate launch /home/x-cli32/chaohan/projects/open-r1/src/open_r1/sftSPLLM.py --config_file "$CFG"