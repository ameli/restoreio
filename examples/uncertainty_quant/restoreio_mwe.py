#! /usr/bin/env python

# Install restoreio with ;#"\texttt{pip install restoreio}"#;
from restoreio import restore

# OpenDap URL of the remote netCDF data
url = 'http://hfrnet-tds.ucsd.edu/thredds/' + \
       'dodsC/HFR/USWC/2km/hourly/RTV/HFRAD' + \
       'AR_US_West_Coast_2km_Resolution_Hou' + \
       'rly_RTV_best.ncd'

# Generate ensemble and reconstruct gaps at ;#$ \OmegaMissing $#;
restore(input=url, output='output.nc',
         min_lon=-122.344, max_lon=-121.781,
         min_lat=36.507, max_lat=36.992,
         time='2017-01-25T03:00:00',
         uncertainty_quant=True, plot=True,
         num_samples=2000, ratio_num_modes=1,
         kernel_width=5, scale_error=0.08,
         detect_land=True, fill_coast=True,
         write_samples=True, verbose=True)
