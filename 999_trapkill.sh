#!/bin/bash


if [ -n "${___TRAPKILL___}"  ]; then


    echo "___TRAKILL___ already set, skip it."

else

    echo "___TRAPKILL___ is not set. Do it now."

    trap 'echo "Exiting... kill jobs now!"; pkill -9 -P $$' SIGINT SIGTERM
    #trap 'echo "Exiting... ready to kill jobs... "; kill $$' EXIT

    ___TRAPKILL___=TRUE
fi
