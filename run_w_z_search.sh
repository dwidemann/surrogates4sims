#!/bin/bash

w=("1" "5" "10" "20" "30" "50" "100" "150")
z=("8" "16" "32" "64" "128" "256")
for k in "${w[@]}"; do
        for p in "${z[@]}"; do
                python -m surrogates4sims.full_svd_surrogate_script --window $k --numComponents $p --gpu_ids 0
        done
done
