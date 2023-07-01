# Install restoreio with ;#"\texttt{pip install restoreio}"#;
from restoreio import restore

# OpenDap URL of the remote netCDF data
url = 'http://hfrnet-tds.ucsd.edu/thredds/' + \
       'dodsC/HFR/USWC/2km/hourly/RTV/HFRAD' + \
       'AR_US_West_Coast_2km_Resolution_Hou' + \
       'rly_RTV_best.ncd'

# Generate ensembles and reconstruct gaps at ;#$ \OmegaMissing $#;
restore(input=url, output='output.nc',
         time='2017-01-25T03:00:00',
         min_lon=-122.344, max_lon=-121.781,
         min_lat=36.507, max_lat=36.992,
         uncertainty_quant=True,
         num_ensembles=2000, ratio_num_modes=1,
         kernel_width=5, scale_error=0.08,
         detect_land=True, fill_coast=True,
         plot=True, verbose=True)
