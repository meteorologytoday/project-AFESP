#!/bin/bash

source 000_setup.sh
source 999_trapkill.sh

params=(
    ecmf CY49R1  0-19
    cwao GEPS8   0-19
)

output_dir=$fig_dir/SST_diff
test_day=2024-12-23

mkdir -p $output_dir


nparams=3
for (( i=0 ; i < $(( ${#params[@]} / $nparams )) ; i++ )); do


    origin="${params[$(( i * $nparams + 0 ))]}"
    model_version="${params[$(( i * $nparams + 1 ))]}"
    ens_range="${params[$(( i * $nparams + 2 ))]}"

    echo ":: origin        = $origin"
    echo ":: model_version = $model_version"
    echo ":: ens_range     = $ens_range"
    
    for lead_day in 0 5 10 20 30 ; do
        
        
        output_file=$output_dir/${origin}_${test_day}_lead-${lead_day}.svg

        if [ -f "$output_file" ] ; then

            echo "Output file $output_file done."

        else
            python3 src/plot_SST_diff.py \
                --archive-root  $S2S_root \
                --origin $origin \
                --model-version $model_version \
                --ens-range $ens_range \
                --nwp-type forecast \
                --start-time $test_day  \
                --lead-day $lead_day \
                --no-display \
                --output $output_file 

        fi

    done

done

wait

echo "Done"
