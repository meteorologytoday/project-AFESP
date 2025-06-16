#!/bin/bash

source 000_setup.sh
source 999_trapkill.sh

params=(
    rjtd CPS3 0-4
#    kwbc CFSv2  0-15
#    ecmf CY49R1  0-19
#    ecmf CY48R1  0-19
#    cwao GEPS8   0-19
)

output_root=$fig_dir/SST_diff


#test_day=2024-12-05
#test_day=2024-12-23
#test_day=2024-06-13

test_days=(
    2024-12-05
    2024-06-13
)


nparams=3
for (( i=0 ; i < $(( ${#params[@]} / $nparams )) ; i++ )); do


    origin="${params[$(( i * $nparams + 0 ))]}"
    model_version="${params[$(( i * $nparams + 1 ))]}"
    ens_range="${params[$(( i * $nparams + 2 ))]}"

    echo ":: origin        = $origin"
    echo ":: model_version = $model_version"
    echo ":: ens_range     = $ens_range"

    for test_day in ${test_days[@]} ; do    
    for verification_dataset in oisst ; do
    for lead_day in 1 20 ; do
        

        output_dir=$output_root/veri_$verification_dataset
        output_file=$output_dir/${origin}_${test_day}_lead-${lead_day}.svg

        if [ -f "$output_file" ] ; then

            echo "Output file $output_file done."

        else

            mkdir -p $output_dir

            python3 src/plot_SST_diff.py \
                --verification-dataset $verification_dataset   \
                --archive-root  $S2S_root      \
                --origin $origin               \
                --model-version $model_version \
                --ens-range $ens_range \
                --nwp-type forecast \
                --start-time $test_day  \
                --lead-day $lead_day \
                --no-display \
                --omit-abs \
                --output $output_file & 

        fi

    done
    done
    done
done

wait

echo "Done"
