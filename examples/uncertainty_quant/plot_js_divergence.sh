#! /usr/bin/bash

# This script plots only the JS divergence result.
#
# Input file:
#     Make sure this file exists: ../data/Monterey_Small_2km_Hourly_2017_01_nc
# Where to run this script:
#     Run this script in /scripts directory.
# What are the outputs:
#     All outputs are stored in /scripts/output_js_divergence directory.

# Make sure the executable "restore" is on the path
export PATH=$PATH:/opt/miniconda3/bin

# Create a subdirectory to store all output files and plots
mkdir -p output_js_divergence
cd output_js_divergence

URL='http://hfrnet-tds.ucsd.edu/thredds/dodsC/HFR/USWC/2km/hourly/RTV/HFRADAR_US_West_Coast_2km_Resolution_Hourly_RTV_best.ncd'

# Produce JS divergence plot (we vary m from 0 to 1 over 100 different points).
# This yields a hundred output files.
num_m=200
for ((i=1; i<=${num_m}; i++))
do
    m=$(bc -l <<< "${i}/${num_m}");
    output=$(printf "output-%03d.nc" ${i})

    echo "Processing ${output}, m=${m}"

    restore -i ${URL} -o ${output} \
        --min-lon -122.344 --max-lon -121.781 \
        --min-lat 36.507 --max-lat 36.9925 \
        --time "2017-01-25T03:00:00" \
        -d 20 -a 20 -s -L 2 -l -u -e 2000 -m ${m}

done

# Return back
cd ..
