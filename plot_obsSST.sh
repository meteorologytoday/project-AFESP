#!/bin/bash

source 000_setup.sh
source 999_trapkill.sh

output_root=$fig_dir/obsSST_diff


params=(
    2023-06-13
)


nparams=1
for (( i=0 ; i < $(( ${#params[@]} / $nparams )) ; i++ )); do


    test_day="${params[$(( i * $nparams + 0 ))]}"

    echo ":: test_day = $test_day"
    python3 src/plot_obsSST_diff.py     \
        --datasets oisst OSTIA_UKMO    \
        --archive-root  $obsSST_root    \
        --beg-date $test_day            \
        --end-date $test_day            \
        --output-root $output_root      \
        --no-display
#        --plot-lat-rng -10 30 \
#        --plot-lon-rng 30 90 \
#        --no-display 


    #parser.add_argument('--plot-lat-rng', type=float, nargs=2, help='Plot range of latitude', default=[-90, 90])
    #parser.add_argument('--plot-lon-rng', type=float, nargs=2, help='Plot range of latitude', default=[0, 360])

done

wait

echo "Done"
