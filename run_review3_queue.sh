#!/bin/sh
# GPU queue for the third review's experiments; one job at a time on the 8 GB card.
export PYTHONIOENCODING=utf-8
F="warn|return F|return torch|attn_output|run_backward"
python element_sensitivity.py Transformer CNN-GlobalPool CNN-Dense 2>&1 | grep -v -i -E "$F"
MATCHED=1 python freq_study.py 2>&1 | grep -v -i -E "$F"
METHODS=baseline,gcr python review2_study.py full DeepSTARR 2>&1 | grep -v -i -E "$F"
python review2_study.py notata DeepSTARR 2>&1 | grep -v -i -E "$F"
python element_sensitivity.py DeepSTARR NT-v2-50M NT-v2-100M NT-v2-500M-LoRA 2>&1 | grep -v -i -E "$F"
echo QUEUE DONE
