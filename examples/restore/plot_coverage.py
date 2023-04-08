#! /usr/bin/env python

# =======
# Imports
# =======

import numpy
import sys
import warnings
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap 
from matplotlib import cm
from mpl_toolkits.basemap import Basemap
import netCDF4

# Modules
import InputOutput
import Geography

# =================
# Make Array Masked
# =================

def MakeArrayMasked(Array):
    """
    Some datasets are not masked, rather they use nan or inf for unavailable datasets.
    This function creates a mask for those datastes.
    """

    if (not hasattr(Array,'mask')) or (Array.mask.size == 1):
        if numpy.isnan(Array).any() or numpy.isinf(Array).any():
            # This array is not masked. Make a mask based no nan and inf
            Mask_nan = numpy.isnan(Array)
            Mask_inf = numpy.isinf(Array)
            Mask = numpy.logical_or(Mask_nan,Mask_inf)
            Array = numpy.ma.masked_array(Array,mask=Mask)
    else:
        # This array is masked. But check if any non-masked value is nan or inf
        for i in range(Array.shape[0]):
            for j in range(Array.shape[1]):
                if Array.mask[i,j] == False:
                    if numpy.isnan(Array[i,j]) or numpy.isinf(Array[i,j]):
                        Array.mask[i,j] = True

    return Array

# =============
# Plot Coverage
# =============

def PlotCoverage(Datetime,Latitude,Longitude,Coverage):

    print("Plot ...")

    # Mesh grid
    LongitudesGrid,LatitudesGrid = numpy.meshgrid(Longitude,Latitude)

    # Corner points (Use 0.05 for MontereyBay and 0.1 for Martha dataset)
    # Percent = 0.05
    Percent = 0.1
    LongitudeOffset = Percent * numpy.abs(Longitude[-1] - Longitude[0])
    LatitudeOffset = Percent * numpy.abs(Latitude[-1] - Latitude[0])

    MinLongitudeWithOffset = numpy.min(Longitude) - LongitudeOffset
    MidLongitude = numpy.mean(Longitude)
    MaxLongitudeWithOffset = numpy.max(Longitude) + LongitudeOffset
    MinLatitudeWithOffset = numpy.min(Latitude) - LatitudeOffset
    MidLatitude = numpy.mean(Latitude)
    MaxLatitudeWithOffset = numpy.max(Latitude) + LatitudeOffset

    # Basemap (set resolution to 'i' for faster rasterization and 'f' for full resolution but very slow.)

    # --------
    # Draw Map
    # --------

    def DrawMap(axis):

        map = Basemap( \
                ax = axis, \
                projection = 'aeqd', \
                llcrnrlon=MinLongitudeWithOffset, \
                llcrnrlat=MinLatitudeWithOffset, \
                urcrnrlon=MaxLongitudeWithOffset, \
                urcrnrlat=MaxLatitudeWithOffset, \
                area_thresh = 0.1, \
                lon_0 = MidLongitude, \
                lat_0 = MidLatitude, \
                resolution='i')

        # Map features
        map.drawcoastlines()
        # map.drawstates()
        # map.drawcountries()
        # map.drawcounties()
        map.drawlsmask(land_color='Linen', ocean_color='#CCFFFF',lakes=True)
        # map.fillcontinents(color='red', lake_color='white', zorder=0)
        map.fillcontinents(color='moccasin')

        # map.bluemarble()
        map.shadedrelief()
        # map.etopo()

        # Latitude and Longitude lines
        LongitudeLines = numpy.linspace(numpy.min(Longitude),numpy.max(Longitude),2)
        LatitudeLines = numpy.linspace(numpy.min(Latitude),numpy.max(Latitude),2)
        # map.drawparallels(LatitudeLines,labels=[1,0,0,0],fontsize=10)
        map.drawmeridians(LongitudeLines,labels=[0,0,0,1],fontsize=10)

        return map

    # Custom colormap from transparent to blue
    Res=100
    # colors = [(0.9,0.54,0.2,c) for c in numpy.linspace(0,1,Res)] # orange
    # colors = [(0.3,0.5,0.1,c) for c in numpy.linspace(0,1,Res)]  # green
    # colors = [(0.17,0.34,0.52,c) for c in numpy.linspace(0,1,Res)]   # blue
    colors = [(0.17,0.4,0.63,c) for c in numpy.linspace(0,1,Res)]   # blue
    cmapblue = mcolors.LinearSegmentedColormap.from_list('mycmap',colors,N=Res)

    # Plot array
    fig,axis = plt.subplots()
    map = DrawMap(axis)

    LongitudesGridOnMap,LatitudesGridOnMap= map(LongitudesGrid,LatitudesGrid)
    # CoverageMap = map.pcolormesh(LongitudesGridOnMap,LatitudesGridOnMap,Coverage,cmap=cm.Blues,edgecolors='blue')
    CoverageMap = map.pcolormesh(LongitudesGridOnMap,LatitudesGridOnMap,Coverage,cmap=cmapblue,vmin=0,vmax=100)
    cbar = fig.colorbar(CoverageMap,ticks=[0,25,50,75,100],orientation='horizontal')

    # FirstTime = netCDF4.num2date(Datetime[0],Datetime.units,Datetime.calendar)
    # LastTime = netCDF4.num2date(Datetime[-1],Datetime.units,Datetime.calendar)
    # axis.set_title('Coverage from: %s to %s'%(FirstTime,LastTime))

    plt.show()

# ====
# Main
# ====

def main(argv):
    """
    This code scans the input file and creates an array of size (lat,log) and initiate it with zseros. Then it counts through all
    time steps that how many times a specific location has mask=False, that is how many times the data are available. Then the array
    (coverage array) is normalized to percent. The result is ploted with basemap.
    """

    # Print usage
    def PrintUsage(ExecName):
        UsageString = "Usage: " + ExecName + "<InputFilename.{nc,ncml}>"
        print(UsageString)

    # Parse arguments, get InputFilename
    InputFilename = ''
    if len(argv) < 1:
        PrintUsage(argv[0])
        sys.exit()
    else:
        InputFilename = argv[1]

    # Check input
    if InputFilename == '':
        PrintUsage(argv[0])
        sys.exit()

    # Open file
    agg = InputOutput.LoadDataset(InputFilename)

    # Load variables
    DatetimeObject,LongitudeObject,LatitudeObject,EastVelocityObject,NorthVelocityObject,EastVelocityErrorObject,NorthVelocityErrorObject = InputOutput.LoadVariables(agg)

    # Get arrays
    Datetime = DatetimeObject
    Longitude = LongitudeObject[:]
    Latitude = LatitudeObject[:]
    U_AllTimes = EastVelocityObject[:]
    V_AllTimes = NorthVelocityObject[:]

    # Sizes
    TimeSize = U_AllTimes.shape[0]
    LatitudeSize = U_AllTimes.shape[1]
    LongitudeSize = U_AllTimes.shape[2]

    # Initialize Inpainted arrays
    FillValue = 999
    Coverage = numpy.ma.zeros((LatitudeSize,LongitudeSize),dtype=float,fill_value=FillValue)

    # Determine the land
    # LandIndices,OceanIndices = Geography.FindLandAndOceanIndices(Longitude,Latitude)
    LandIndices,OceanIndices = Geography.FindLandAndOceanIndices2(Longitude,Latitude)

    # Iterate through all times
    print("Calculating coverage map ...")
    for TimeIndex in range(TimeSize):

        # print("Process: %d / %d"%(TimeIndex+1,TimeSize))

        # Get U and V
        U = U_AllTimes[TimeIndex,:]
        V = V_AllTimes[TimeIndex,:]

        # Mask U and V (if they are not masked already)
        U = MakeArrayMasked(U)
        V = MakeArrayMasked(V)

        # Iterate over lon and lat
        for LatitudeIndex in range(LatitudeSize):
            for LongitudeIndex in range(LongitudeSize):

                MaskU = U.mask[LatitudeIndex,LongitudeIndex]
                MaskV = V.mask[LatitudeIndex,LongitudeIndex]

                if (MaskU == False) and (MaskV == False):
                    Coverage[LatitudeIndex,LongitudeIndex] = Coverage[LatitudeIndex,LongitudeIndex] + 1.0

    # Normalize coverage to percent
    Coverage = 100.0 * Coverage / TimeSize

    # Mask land indices
    if LandIndices.shape[0] > 0:
        for i in range(LandIndices.shape[0]):
            Coverage[LandIndices[i,0],LandIndices[i,1]] = numpy.ma.masked

    # Plot
    PlotCoverage(Datetime,Latitude,Longitude,Coverage)

    agg.close()

# ===========
# System Main
# ===========

if __name__ == "__main__":

    # Converting all warnings to error
    # warnings.simplefilter('error',UserWarning)

    # Main function
    main(sys.argv)
