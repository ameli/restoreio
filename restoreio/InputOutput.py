# SPDX-FileCopyrightText: Copyright 2016, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the license found in the LICENSE.txt file in the root directory
# of this source tree.

# =======
# Imports
# =======

import netCDF4
import pyncml
import os.path
import getopt
import sys
import textwrap
import numpy
import time
import datetime
try:
    # Python 3
    from urllib.parse import urlparse
except ImportError:
    # python 2
    from urlparse import urlparse

# ===============
# Parse Arguments
# ===============

def ParseArguments(argv):
    """
    Parses the argument of the executable and obtains the filename.

    Input file is netcdf nc file or ncml file.
    Output file is nc file.

    Options:
    -i --input:        required,   full path of input nc file or ncml file
    -o --output:       required,   full path of nc file
    -d --diffusivity,  optional,   the diffusivity followed by a real positive number like 20
    -s --sweep,        optional,   sweeps the inpainting of image in various directions
    -p --plot,         optional,   if -p used, the time frame after -p is plotted. no computation is done for other time frames.
    """

    # -------------
    # Print Version
    # -------------

    def PrintVersion():

        VersionString = \
                """
Version 0.0.1
Siavash Ameli
University of California, Berkeley
                """

        print(VersionString)

    # -----------
    # Print Usage
    # -----------

    def PrintUsage(ExecName):
        UsageString = "Usage: " + ExecName + " -i <InputFilename.{nc, ncml}> -o <OutputFilename.nc> [options]"
        OptionsString = \
        """
Required arguments:

    -i --input                Input filename. This should be the full path.
                              Input file extension can be *.nc or *.ncml.
    -o --output               Output filename. This should be full path. Output
                              file extension should be *.nc file.
                              If option '-p' is specified with positive time
                              frame, the '-o' option is not required.

Optional arguments:

    -h --help                 Prints this help message.
    -v --version              Prints the version and author info.
    -d --difusivity[=float]   Diffusivity of the PDE solver (real number).
                              Large number: diffusion dominant,
                              Small numbers: Advection dominant.
    -s --sweep                Sweeps the PDE solution in all flipped directions
                              of image. This ensures an even solution
                              independent of direction. 
    -p --plot                 Instead of iterating through all time frames,
                              only solves the given time frame in option -t,
                              and plots results. If uncertainty quantification
                              is enabled with -u, the plot option will plot
                              figures for the statistical analysis for the time
                              frame in  option -t.
    -L --exclude-land[=int]   Enables/disables the exclusion of land an ocean.
                              Otherwise the entire domain is treated as ocean.
                              - L 0: Does not exclude land from ocean. All data
                                are treated as in the ocean.
                              - L 1: Excludes ocean and land. Most accurate,
                                slowest.
                              - L 2: Excludes ocean and land. Less accurate,
                                fastest.
                              - L 3: Excludes ocean and land. Currently this
                                option is not working.
    -l --include-nearshore    Includes the ocean area between data domain
                              (convex/concave hull) and the shore. This fills
                              the gap up to the coast. This is only effective
                              if '-L' is used so that the land is separated to
                              the ocean.
    -c --convex               Instead of using concave hull (alpha shape), this
                              options uses convex hull of the area.
    -a --alpha[=float]        The alpha number for alpha shape. If not
                              specified or a negative number, it is computed.
                              This option is only for concave shapes. Hence
                              '-a, --alpha' is ineffective if using '-c'.
    -r --refine[=int]         Refines the grid. Int set to 1, this is the
                              original grid without refinement. 2, 3, etc
                              refines grid in each axes. Default is 1.
    -t --time-frame[=int]     The time to plot or process the uncertainty
                              quantification. Default is -1, the last time.
    -u --uncertainty          Enables uncertainty quantification for the time
                              frame in option -t. This either produces
                              results in output file in option -o, or plots
                              results with option -p.
    -e --num-ensembles[=int]  Number of ensembles used for uncertainty
                              quantification. Default is 1000.
    -m --min-file             Minimum file iterator string, to be used for
                              processing multiple files, like -m 003 -n 012
                              If this option is used, the -n or --max-file
                              option should also be used.
    -n --max-file             Maximum file iterator string, to be used for
                              processing multiple files, like -m 003 -n 012
                              If this option is used, the -m or --min-file
                              option should also be used.
                """
        ExampleString = \
                """
Examples:

    1. Martha's Vineyard dataset, ncml files (uses series of nc files), using convex hull with diffusivity 20 and sweep, not filling the
       gap up to the coast (not including the land). Since it is convex shape, we do not specify alpha. The output is an *.nc file
       containing all inpainted time frames of the ncml files.

       $ %s -i /media/Data/agg_HFR.ncml -o ~/HFR_restored.nc -d 20 -s -c -L 0

    2. Same setting as above, except we only plot the time frame 20 without going through all time frames. No output file is written,
       only the results are ploted. 

       $ %s -i /media/Data/agg_HFR.ncml -d 20 -s -c -L 0 -p -t 20

    3. Monterey Bay dataset, one *.nc input file, using concave hull with alpha 10, diffusivity 20 and sweep.
       We separate ocean with the land (if land exists) and only inpaint areas in ocean by using option '-L'.
       We only plot one time frame at frame 102 without processing other time frames.

       $ %s -i /media/Data/MontereyBay_2km.nc -d 20 -s -L 1 -a 10 -p -t 102

    4. Same as above. But we not only exclude the land from the ocean (option -L), also we extend the inpainting up to the 
       coast line by including the land to the concave hull (option -l)

       $ %s -i /media/Data/MontereyBay_2km.nc -d 20 -s -L 1 -l -a 10 -p -t 102

    5. Same as above without plotting, but going through all time steps and write to output *.nc file, also with refinement

       $ %s -i /media/Data/MontereyBay_2km.nc -o ~/MontereyBay_2km_restored.nc -d 20 -s -L 1 -l -a 10 -r 2

    6. Uncertainty quantification with 2000 ensembles, plotting (no output file), at timeframe 102

       $ %s -i /media/Data/MontereyBay_2km.nc -d 20 -s -L 1 -l -a 10 -t 102 -u -e 2000 -p

    7. Processing multiple sepaate files. Suppose we have these input files:
            /home/sia/Ensembles/File001.nc
            /home/sia/Ensembles/File002.nc
            ...
            /home/sia/Ensembles/File012.nc

       and we want to store them all in this output file:
            /home/sia/Ensembles-Restored/OutputFile.zip

       For uncertanty quantification: 
       $ %s -i /home/sia/Ensembles/File001.nc -o /home/sia/Ensembles-Restored/OutputFile.zip -d 20 -s -L 1 -l -a 10 -t 102 -u -e 2000 -m 001 -n 012

       For resotration
       $ %s -i /home/sia/Ensembles/File001.nc -o /home/sia/Ensembles-Restored/OutputFile.zip -d 20 -s -L 1 -l -a 10 -t 102 -m 001 -n 012

                """%(ExecName, ExecName, ExecName, ExecName, ExecName, ExecName, ExecName, ExecName)

        print(UsageString)
        print(OptionsString)
        print(ExampleString)
        PrintVersion()

    # -----------------

    # Initialize variables (defaults)
    Arguments = \
    {
        'FullPathInputFilename': '',
        'FullPathOutputFilename': '',
        'Diffusivity': 20,
        'SweepAllDirections': False,
        'Plot': False,
        'ExcludeLandFromOcean': 0,
        'IncludeLandForHull': False,
        'UseConvexHull': False,
        'Alpha': -1,
        'RefinementLevel': 1,
        'TimeFrame': -1,
        'UncertaintyQuantification': False,
        'NumEnsembles': 1000,
        "ProcessMultipleFiles": False,
        "MultipleFilesMinIteratorString": '',
        "MultipleFilesMaxIteratorString": ''
    }

    # Get options
    try:
        opts, args = getopt.getopt(argv[1:], "hvi:o:d:psL:lca:r:ut:e:m:n:", \
                ["help", "version", "input=", "output=", "diffusivity=", "plot", "sweep", "exclude-land=", "include-nearshore", "convex", "alpha=", "refine=", "uncertainty", "time-frame=", "num-ensembles=", "min-file=", "max-file="])
    except getopt.GetoptError:
        PrintUsage(argv[0])
        sys.exit(2)

    # Assign options
    for opt, arg in opts:
        
        if opt in ('-h', '--help'):
            PrintUsage(argv[0])
            sys.exit()
        elif opt in ('-v', '--version'):
            PrintVersion()
            sys.exit()
        elif opt in ("-i", "--input"):
            Arguments['FullPathInputFilename'] = arg
        elif opt in ("-o", "--output"):
            Arguments['FullPathOutputFilename'] = arg
        elif opt in ('-d', '--diffusivity'):
            Arguments['Diffusivity'] = float(arg)
        elif opt in ('-s', '--sweep'):
            Arguments['SweepAllDirections'] = True
        elif opt in ('-p', '--plot'):
            Arguments['Plot'] = True
        elif opt in ('-L', '--exclude-land'):
            Arguments['ExcludeLandFromOcean'] = int(arg)
        elif opt in ('-l', '--include-nearshore'):
            Arguments['IncludeLandForHull'] = True
        elif opt in ('-c', '--convex'):
            Arguments['UseConvexHull'] = True
        elif opt in ('-a', '--alpha'):
            Arguments['Alpha'] = float(arg)
        elif opt in ('-r', '--refine'):
            Arguments['RefinementLevel'] = int(arg)
        elif opt in ('-u', '--uncertainty'):
            Arguments['UncertaintyQuantification'] = True
        elif opt in ('-t', '--time-frame'):
            Arguments['TimeFrame'] = int(arg)
        elif opt in ('-e', '--num-ensembles'):
            Arguments['NumEnsembles'] = int(arg)
        elif opt in ('-m', '--min-file'):
            Arguments['MultipleFilesMinIteratorString'] = arg
        elif opt in ('-n', '--max-file'):
            Arguments['MultipleFilesMaxIteratorString'] = arg

    # Check Arguments
    if len(argv) < 2:
        PrintUsage(argv[0])
        sys.exit()

    # Check InputFilename
    if (Arguments['FullPathInputFilename'] == ''):
        PrintUsage(argv[0])
        print(' ')
        print('Error: No input file is selected with option (-i).')
        sys.exit(2)

    # We can not have empty outputfilename and not plotting.
    if (Arguments['FullPathOutputFilename'] == '') and (Arguments['Plot'] == False):
        PrintUsage(argv[0])
        print(' ')
        print('ERROR: Either the output file should be specified (-o) or the plot option should be enabled (-p).')
        sys.exit(2) 

    # Check include land
    if Arguments['ExcludeLandFromOcean'] == 0:
        Arguments['IncludeLandForHull'] = False

    # Check Processing multiple file
    if (Arguments['MultipleFilesMinIteratorString'] != '') and (Arguments['MultipleFilesMaxIteratorString'] != ''):
        if (Arguments['MultipleFilesMinIteratorString'] == '') or (Arguments['MultipleFilesMaxIteratorString'] == ''):
            raise ValueError('To process multiple files, both min and max file iterator should be specified.')
        else:
            Arguments['ProcessMultipleFiles'] = True

    return Arguments

# ==================
# Load Local Dataset
# ==================

def LoadLocalDataset(Filename):
    """
    Opens either ncml or nc file and returns the aggregation file object.
    """

    print("Message: Loading data ... ")
    sys.stdout.flush()

    # Check file extenstion
    FileExtension = os.path.splitext(Filename)[1]
    if FileExtension == '.ncml':

        # Change directory
        DataDirectory = os.path.dirname(Filename)
        CurrentDirectory = os.getcwd()
        os.chdir(DataDirectory)

        # NCML
        try:
            NCMLString = open(Filename, 'r').read()
            NCMLString = NCMLString.encode('ascii')
            ncml = pyncml.etree.fromstring(NCMLString)
            nc = pyncml.scan(ncml=ncml)

            # Get nc files list
            FilesList = [ f.path for f in nc.members ]
            os.chdir(CurrentDirectory)

            # Aggregate
            agg = netCDF4.MFDataset(FilesList, aggdim='t')

        except:
            print('ERROR: Can not read local ncml file: ' + Filename)
            raise

        return agg

    elif FileExtension == '.nc':

        try:
            nc = netCDF4.Dataset(Filename)
        except:
            print('ERROR: Can not read local netcdf file: ' + Filename)
            raise

        return nc

    else:
        raise ValueError("File should be either *.ncml or *.nc.")

# ===================
# Load Remote Dataset
# ===================

def LoadRemoteDataset(URL):
    """
    URL can be point to a *.nc or *.ncml file.
    """

    try:
        nc = netCDF4.Dataset(URL)
    except:
        print('ERROR: Can not read remote file: ' + URL)
        raise

    return nc

# ============
# Load Dataset
# ============

def LoadDataset(InputFilename):
    """
    Dispatches the execution to either of the following two functions:
    1. LoadLocalDataset: For files where the InputFilename is a path on the local machine.
    2. LoadRemoteDataset: For files remotely where InputFilename is a URL.
    """

    # Check if the InputFilename has a "host" name
    if bool(urlparse(InputFilename).netloc):
        # InputFilename is a URL
        return LoadRemoteDataset(InputFilename)
    else:
        # InputFilename is a path
        return LoadLocalDataset(InputFilename)

# ==============
# Load Variables
# ==============

def LoadVariables(agg):
    """
    Finds the following variables from the aggregation object agg.

    - Time
    - Longitude
    - Latitude
    - Eastward velocity U
    - Northward velocity V
    """

    # ---------------
    # Search Variable
    # ---------------

    def SearchVariable(agg, NamesList, StandardNamesList):
        """
        This function searches for a list of names and standard names to match a variable.

        Note: All strings are compared with their lowercase form.
        """

        VariableFound = False

        # Search among standard names list
        for StandardName in StandardNamesList:
            for Key in agg.variables.keys():
                Variable = agg.variables[Key]
                if hasattr(Variable, 'standard_name'):
                    StandardNameInAgg = Variable.standard_name
                    if StandardName.lower() == StandardNameInAgg.lower():
                        Object = agg.variables[Key]
                        VariableFound = True
                        break
            if VariableFound == True:
                break

        # Search among names list
        if VariableFound == False:
            for Name in NamesList:
                for Key in agg.variables.keys():
                    if Name.lower() == Key.lower():
                        Object = agg.variables[Key]
                        VariableFound = True
                        break
                if VariableFound == True:
                    break

        # Lat check to see if the variable is found
        if VariableFound == False:
            Object = None

        return Object

    # -------------

    # Time
    TimeNamesList = ['time', 'datetime', 't']
    TimeStandardNamesList = ['time']
    DatetimeObject = SearchVariable(agg, TimeNamesList, TimeStandardNamesList)
    if DatetimeObject == None:
        raise RuntimeError('Time object can not be found in netCDF file.')

    # Longitude
    LongitudeNamesList = ['longitude', 'lon', 'long']
    LongitudeStandardNamesList = ['longitude']
    LongitudeObject = SearchVariable(agg, LongitudeNamesList, LongitudeStandardNamesList)
    if LongitudeObject == None:
        raise RuntimeError('Longitude object can not be found in netCDF file.')

    # Latitude
    LatitudeNamesList = ['latitude', 'lat']
    LatitudeStandardNamesList = ['latitude']
    LatitudeObject = SearchVariable(agg, LatitudeNamesList, LatitudeStandardNamesList)
    if LatitudeObject == None:
        raise RuntimeError('Latitude object can not be found in netCDF file.')

    # East Velocity
    EastVelocityNamesList = ['east_vel', 'eastward_vel', 'u', 'east_velocity', 'eastward_velocity']
    EastVelocityStandardNamesList = ['surface_eastward_sea_water_velocity', 'eastward_sea_water_velocity']
    EastVelocityObject = SearchVariable(agg, EastVelocityNamesList, EastVelocityStandardNamesList)
    if EastVelocityObject == None:
        raise RuntimeError('EastVelocity object can not be found in netCDF file.')

    # North Velocity
    NorthVelocityNamesList = ['north_vel', 'northward_vel', 'v', 'north_velocity', 'northward_velocity']
    NorthVelocityStandardNamesList = ['surface_northward_sea_water_velocity', 'northward_sea_water_velocity']
    NorthVelocityObject = SearchVariable(agg, NorthVelocityNamesList, NorthVelocityStandardNamesList)
    if NorthVelocityObject == None:
        raise RuntimeError('NorthVelocity object can not be found in netCDF file.')

    # East Velocity Error
    EastVelocityErrorNamesList = ['east_err', 'dopx']
    EastVelocityErrorStandardNamesList = []
    EastVelocityErrorObject = SearchVariable(agg, EastVelocityErrorNamesList, EastVelocityErrorStandardNamesList)

    # North Velocity Error
    NorthVelocityErrorNamesList = ['north_err', 'dopy']
    NorthVelocityErrorStandardNamesList = []
    NorthVelocityErrorObject = SearchVariable(agg, NorthVelocityErrorNamesList, NorthVelocityErrorStandardNamesList)

    return DatetimeObject, LongitudeObject, LatitudeObject, EastVelocityObject, NorthVelocityObject, EastVelocityErrorObject, NorthVelocityErrorObject

# =================
# Prepare Datetimes
# =================

def PrepareDatetimes(TimeIndices, DatetimeObject):
    """
    This is used in writer function.
    Converts date char format to datetime numeric format.
    This parses the times chars and converts them to date times.
    """

    # Datetimes units
    if (hasattr(DatetimeObject, 'units')) and (DatetimeObject.units != ''):
        DatetimesUnit = DatetimeObject.units
    else:
        DatetimesUnit = 'days since 1970-01-01 00:00:00 UTC'

    # Datetimes calendar
    if (hasattr(DatetimeObject, 'calendar')) and (DatetimeObject.calendar != ''):
        DatetimesCalendar = DatetimeObject.calendar
    else:
        DatetimesCalendar ='gregorian'

    # Datetimes
    DaysList = []
    OriginalDatetimes = DatetimeObject[:]

    if OriginalDatetimes.ndim == 1:

        # Datetimes in original dataset is already suitable to use
        Datetimes = OriginalDatetimes[TimeIndices]

    elif OriginalDatetimes.ndim == 2:

        # Datetime in original dataset is in the form of string. They should be converted to numerics
        # for i in range(OriginalDatetimes.shape[0]):

        # If TimeIndices is just a single number, convert it to list.
        if type(TimeIndices) == int:
            TimeIndices = [TimeIndices]

        # Iteration over time indices
        for i in TimeIndices:

            # Get row as string (often it is already a string, or a byte type)
            CharTime = numpy.chararray(OriginalDatetimes.shape[1])
            for j in range(OriginalDatetimes.shape[1]):
                CharTime[j] = OriginalDatetimes[i, j].astype('str')

            # Parse chars to integers

            # Year
            if CharTime.size >= 4:
                Year = int(CharTime[0] + CharTime[1] + CharTime[2] + CharTime[3])
            else:
                Year = 1970

            # Month
            if CharTime.size >= 6:
                Month = int(CharTime[5] + CharTime[6])
            else:
                Month = 1

            # Day
            if CharTime.size >= 9:
                Day = int(CharTime[8] + CharTime[9])
            else:
                Day = 1

            # Hour
            if CharTime.size >= 13:
                Hour = int(CharTime[11] + CharTime[12])
            else:
                Hour = 0
            
            # Monute
            if CharTime.size >= 15:
                Minute = int(CharTime[14] + CharTime[15])
            else:
                Minute = 0

            # Second
            if CharTime.size >= 19:
                Second = int(CharTime[17] + CharTime[18])
            else:
                Second = 0

            # Create Day object
            DaysList.append(datetime.datetime(Year, Month, Day, Hour, Minute, Second))

        # Convert dates to numbers
        Datetimes = netCDF4.date2num(DaysList, units=DatetimesUnit, calendar=DatetimesCalendar)
    else:
        raise RuntimeError("Datetime ndim is more than 2.")

    return Datetimes, DatetimesUnit, DatetimesCalendar

# =================
# Write Output File
# =================

def WriteOutputFile( \
        TimeIndices, \
        DatetimeObject, \
        Longitude, \
        Latitude, \
        MaskInfo, \
        U_AllTimes_Inpainted, \
        V_AllTimes_Inpainted, \
        U_AllTimes_Inpainted_Error, \
        V_AllTimes_Inpainted_Error, \
        FillValue, \
        OutputFilename):
    """
    Writes the inpainted array to an output netcdf file.
    """

    print("Message: Writing to NetCDF file ...")
    sys.stdout.flush()
 
    OutputFile = netCDF4.Dataset(OutputFilename, 'w', format='NETCDF4_CLASSIC')

    # Dimensions
    OutputFile.createDimension('time', None)
    OutputFile.createDimension('lon', len(Longitude))
    OutputFile.createDimension('lat', len(Latitude))

    # Prepare times from file
    Datetimes, DatetimeUnit, DatetimeCalendar = PrepareDatetimes(TimeIndices, DatetimeObject)

    # Datetime
    OutputDatetime = OutputFile.createVariable('time', numpy.dtype('float64').char, ('time', ))
    OutputDatetime[:] = Datetimes
    OutputDatetime.units = DatetimeUnit
    OutputDatetime.calendar = DatetimeCalendar
    OutputDatetime.standard_name = 'time'
    OutputDatetime._CoordinateAxisType = 'Time'
    OutputDatetime.axis = 'T'

    # Longitude
    OutputLongitude = OutputFile.createVariable('lon', numpy.dtype('float64').char, ('lon', ))
    OutputLongitude[:] = Longitude
    OutputLongitude.units = 'degree_east'
    OutputLongitude.standard_name = 'longitude'
    OutputLongitude.positive = 'east'
    OutputLongitude._CoordinateAxisType = 'Lon'
    OutputLongitude.axis = 'X'
    OutputLongitude.coordsys = 'geographic'

    # Latitude
    OutputLatitude = OutputFile.createVariable('lat', numpy.dtype('float64').char, ('lat', ))
    OutputLatitude[:] = Latitude
    OutputLatitude.units = 'degree_north'
    OutputLatitude.standard_name = 'latitude'
    OutputLatitude.positive = 'up'
    OutputLatitude._CoordinateAxisType = 'Lat'
    OutputLatitude.axis = 'Y'
    OutputLatitude.coordsys = 'geographic'

    # Mask Info
    Mask = OutputFile.createVariable('Mask', numpy.dtype('float64').char, ('time', 'lat', 'lon', ), fill_value=FillValue, zlib=True)
    Mask[:] = MaskInfo
    Mask.long_name = "Integer values at each points. \n \
            -1: Indicates points on land. These points are not used. \n \
            0:  Indicates points in ocean with valid velocity data. These points are used for restoration. \n \
            1: Indicates points in ocean inside convex/concave hull of data domain but with missing velocity data. These points are restored. \n \
            2: Indicates points in ocean outside convex/concave hull of data domain but with missing velocity data. These points are not used."
    Mask.coordinates = 'Longitude Latitude datetime'
    Mask.missing_value = FillValue
    Mask.coordsys = "geographic"

    # Velocity U
    OutputU = OutputFile.createVariable('East_vel', numpy.dtype('float64').char, ('time', 'lat', 'lon', ), fill_value=FillValue, zlib=True)
    OutputU[:] = U_AllTimes_Inpainted
    OutputU.units = 'm s-1'
    OutputU.standard_name = 'surface_eastward_sea_water_velocity'
    OutputU.positive = 'toward east'
    OutputU.coordinates = 'Longitude Latitude datetime'
    OutputU.missing_value = FillValue
    OutputU.coordsys = "geographic"

    # Velocity V
    OutputV = OutputFile.createVariable('North_vel', numpy.dtype('float64').char, ('time', 'lat', 'lon', ), fill_value=FillValue, zlib=True)
    OutputV[:] = V_AllTimes_Inpainted
    OutputV.units = 'm s-1'
    OutputV.standard_name = 'surface_northward_sea_water_velocity'
    OutputV.positive = 'toward north'
    OutputV.coordinates = 'Longitude Latitude datetime'
    OutputV.missing_value = FillValue
    OutputV.coordsys = "geographic"

    # Velocity U Error
    if U_AllTimes_Inpainted_Error is not None:
        OutputUError = OutputFile.createVariable('East_err', numpy.dtype('float64').char, ('time', 'lat', 'lon', ), fill_value=FillValue, zlib=True)
        OutputUError[:] = U_AllTimes_Inpainted_Error
        OutputUError.units = 'm s-1'
        OutputUError.positive = 'toward east'
        OutputUError.coordinates = 'Longitude Latitude datetime'
        OutputUError.missing_value = FillValue
        OutputUError.coordsys = "geographic"

    # Velocity V Error
    if V_AllTimes_Inpainted_Error is not None:
        OutputVError = OutputFile.createVariable('North_err', numpy.dtype('float64').char, ('time', 'lat', 'lon', ), fill_value=FillValue, zlib=True)
        OutputVError[:] = V_AllTimes_Inpainted_Error
        OutputVError.units = 'm s-1'
        OutputVError.positive = 'toward north'
        OutputVError.coordinates = 'Longitude Latitude datetime'
        OutputVError.missing_value = FillValue
        OutputVError.coordsys = "geographic"

    # Global Attributes
    OutputFile.Conventions = 'CF-1.6'
    OutputFile.COORD_SYSTEM = 'GEOGRAPHIC'
    OutputFile.contributor_name = 'Siavash Ameli'
    OutputFile.contributor_email = 'sameli@berkeley.edu'
    OutputFile.contributor_role = 'Post process data to fill missing points.'
    OutputFile.institution = 'University of California, Berkeley'
    OutputFile.date_modified = time.strftime("%x")
    OutputFile.title = 'Restored missing data inside the data domain'
    OutputFile.source = 'Surface observation using high frequency radar.'
    OutputFile.summary = """The HFR original data contain missing data points both inside and outside the computational domain.
            The missing points that are inside a convex hull around the domain of available valid data points are filled. This technique uses a
            PDE based video restoration."""
    OutputFile.project = 'Advanced Lagrangian Predictions for Hazards Assessments (NSF-ALPHA)'
    OutputFile.acknowledgement = 'This material is based upon work supported by the National Science Foundation Graduate Research Fellowship under Grant No. 1520825.'
    OutputFile.geospatial_lat_min = "%f"%(numpy.min(Latitude[:]))
    OutputFile.geospatial_lat_max = "%f"%(numpy.max(Latitude[:]))
    OutputFile.geospatial_lat_units = 'degree_north'
    OutputFile.geospatial_lon_min = "%f"%(numpy.min(Longitude[:]))
    OutputFile.geospatial_lon_max = "%f"%(numpy.max(Longitude[:]))
    OutputFile.geospatial_lon_units = 'degree_east'
    OutputFile.geospatial_vertical_min = '0'
    OutputFile.geospatial_vertical_max = '0'

    OutputFile.time_coverage_start = "%s"%(netCDF4.num2date(OutputDatetime[0], units=OutputDatetime.units, calendar=OutputDatetime.calendar))
    OutputFile.time_coverage_end = "%s"%(netCDF4.num2date(OutputDatetime[-1], units=OutputDatetime.units, calendar=OutputDatetime.calendar))
    OutputFile.cdm_data_type = 'grid'
 
    # Close streams
    OutputFile.close()

    print("Wrote to: %s."%OutputFilename)
    print("Message: Writing to NetCDF file ... Done.")
    sys.stdout.flush()
