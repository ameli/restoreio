#! /usr/bin/env python

# =======
# Imports
# =======

import numpy
import sys
import time
import warnings
import netCDF4
import pyncml
import os.path
import getopt
import textwrap
import datetime
import json
from pydap.client import open_url

try:
    # For python 3
    from urllib.parse import urlparse
except ImportError:
    # For python 2
    from urlparse import urlparse

# ====================
# Terminate With Error
# ====================

def TerminateWithError(Message):
    """
    Returns an incomplete Json object with ScanStatus = False, and an error message.
    """

    # Fill output with defaults
    DatasetInfoDict = \
    {
        "Scan": \
        {
            "ScanStatus": False,
            "Message": Message
        }
    }

    # Print out and exit gracefully
    DatasetInfoJson = json.dumps(DatasetInfoDict,indent=4)
    print(DatasetInfoJson)
    sys.stdout.flush()
    sys.exit()

# ===============
# Parse Arguments
# ===============

def ParseArguments(argv):
    """
    Parses the argument of the executable and obtains the filename.

    Input file is netcdf nc file or ncml file.
    Output is a strigified Json.
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
        UsageString = "Usage: " + ExecName + " -i <InputFilename.{nc,ncml}>"
        OptionsString = \
        """
Required arguments:

    -i --input          Input filename. This should be the full path. Input file extension can be *.nc or *.ncml.

Optional arguments:

    -h --help                   Prints this help message.
    -v --version                Prints the version and author info.
    -V --Velocity               Scans Velocities in the dataset.
                """
        ExampleString = \
                """
Examples:

    1. Using a url on a remote machine. This will not scan the velocities.
       $ %s -i <url>

    2. Using a filename on the local machine. This will not scan the velocities
       $ %s -i <filename>

    3. Scan velocities form a url or filename:
       $ %s -V -i <url or filename>
                """%(ExecName,ExecName,ExecName)

        print(UsageString)
        print(OptionsString)
        print(ExampleString)
        PrintVersion()

    # -----------------

    # Initialize variables (defaults)
    InputFilename = ''
    ScanVelocityStatus = False

    try:
        opts,args = getopt.getopt(argv[1:],"hvVi:",["help","version","Velocity","input="])
    except getopt.GetoptError:
        PrintUsage(argv[0])
        sys.exit(2)

    # Assign options
    for opt,arg in opts:
        
        if opt in ('-h','--help'):
            PrintUsage(argv[0])
            sys.exit()
        elif opt in ('-v','--version'):
            PrintVersion()
            sys.exit()
        elif opt in ("-i","--input"):
            InputFilename = arg
        elif opt in ("-V","--Velocity"):
            ScanVelocityStatus = True

    # Check Arguments
    if len(argv) < 2:
        PrintUsage(argv[0])
        sys.exit(0)

    # Check InputFilename
    if (InputFilename == ''):
        PrintUsage(argv[0])
        raise RuntimeError("InputFilename is empty.")

    return InputFilename,ScanVelocityStatus

# ==================
# Load Local Dataset
# ==================

def LoadLocalDataset(Filename):
    """
    Opens either ncml or nc file and returns the aggregation file object.
    """

    # Check file extenstion
    FileExtension = os.path.splitext(Filename)[1]
    if FileExtension == '.ncml':

        # Change directory
        DataDirectory = os.path.dirname(Filename)
        CurrentDirectory = os.getcwd()
        os.chdir(DataDirectory)

        # NCML
        NCMLString = open(Filename,'r').read()
        NCMLString = NCMLString.encode('ascii')
        ncml = pyncml.etree.fromstring(NCMLString)
        nc = pyncml.scan(ncml=ncml)

        # Get nc files list
        FilesList = [ f.path for f in nc.members ]
        os.chdir(CurrentDirectory)

        # Aggregate
        agg = netCDF4.MFDataset(FilesList,aggdim='t')
        return agg

    elif FileExtension == '.nc':

        nc = netCDF4.Dataset(Filename)
        return nc

    else:
        raise RuntimeError("File should be either ncml or nc. Filename: " + Filename)

# ===================
# Load Remote Dataset
# ===================

def LoadRemoteDataset(URL):
    """
    URL can be point to a *.nc or *.ncml file.
    """

    # Check URL is opendap
    if (URL.startswith('http://') == False) and (URL.startswith('https://') == False):
        TerminateWithError('Input data URL does not seem to be a URL. A URL should start with <code>http://</code> or <code>https://</code>.')
    elif ("/thredds/dodsC/" not in URL) and ("opendap" not in URL):
        TerminateWithError("Input data URL is not an <b>OpenDap</b> URL or is not hosted on a THREDDs server. Check if your data URL contains <code>/thredds/dodsC/</code> or <code>/opendap/</code>.")
    elif URL.endswith(('.nc','.nc.gz','.ncd','.ncml','.ncml.gz')) == False:
        TerminateWithError('The input data URL is not a <i>netcdf</i> file. The URL should end with <code>.nc</code>, <code>.nc.gz</code>, <code>.ncml</code> or <code>.ncml.gz</code>.')

    # Check file extenstion
    FileExtension = os.path.splitext(URL)[1]

    # Case of zipped files (get the correct file extension before the '.gz')
    if FileExtension == ".gz":
        FileExtension = os.path.splitext(URL[:-3])[1]

    if FileExtension == '.ncml':
        
        # nc = open_url(URL)
        nc = netCDF4.Dataset(URL)

    elif (FileExtension == '.nc') or (FileExtension == '.ncd'):

        nc = netCDF4.Dataset(URL)

    else:
        TerminateWithError('File extension in the data URL should be either <code>.nc</code>, <code>.nc.gz</code>, <code>.ncml</code> or <code>.ncml.gz</code>.')

    return nc

# ============
# Load Dataset
# ============

def LoadDataset(InputFilename):
    """
    Dispatches the execution to either of the following two functions:
    1. LoadMultiFileDataset: For files where the InputFilename is apath on the local machine.
    2. LoadRemoteDataset: For files remotely where InputFilename is a URL.
    """

    # Check inpit filename
    if InputFilename == '':
        TerminateWithError('Input data URL is empty. You should provide an OpenDap URL.')

    # Check if the InputFilename has a "host" name
    if bool(urlparse(InputFilename).netloc):
        # InputFilename is a URL
        return LoadRemoteDataset(InputFilename)
    else:
        # InputFilename is a path
        return LoadLocalDataset(InputFilename)

# ===============
# Search Variable
# ===============

def SearchVariable(agg,NamesList,StandardNamesList):
    """
    This function searches for a list of names and standard names to match a variable.

    Note: All strings are compared with their lowercase form.
    """

    VariableFound = False
    ObjectName = ''
    ObjectStandardName = ''

    # Search among standard names list
    for StandardName in StandardNamesList:
        for Key in agg.variables.keys():
            Variable = agg.variables[Key]
            if hasattr(Variable,'standard_name'):
                StandardNameInAgg = Variable.standard_name
                if StandardName.lower() == StandardNameInAgg.lower():
                    Object = agg.variables[Key]
                    ObjectName = Object.name
                    ObjectStandardName = Object.standard_name
                    VariableFound = True
                    break
        if VariableFound == True:
            break

    # Search among names list
    if VariableFound == False:
        for Name in NamesList + StandardNamesList:
            for Key in agg.variables.keys():
                if Name.lower() == Key.lower():
                    Object = agg.variables[Key]
                    ObjectName = Object.name
                    if hasattr(Object,'standard_name'):
                        ObjectStandardName = Object.standard_name
                    VariableFound = True
                    break
            if VariableFound == True:
                break

    # Last check to see if the variable is found
    if VariableFound == False:
        return None

    return Object,ObjectName,ObjectStandardName

# =============================
# Load Time And Space Variables
# =============================

def LoadTimeAndSpaceVariables(agg):
    """
    Finds the following variables from the aggregation object agg.

    - Time
    - Longitude
    - Latitude
    """

    # Time
    TimeNamesList = ['time','datetime','t']
    TimeStandardNamesList = ['time']
    DatetimeObject,DatetimeName,DatetimeStandardName = SearchVariable(agg,TimeNamesList,TimeStandardNamesList)

    # Check time variable
    if DatetimeObject is None:
        TerminateWithError('Can not find the <i>time</i> variable in the netcdf file.')
    elif hasattr(DatetimeObject,'units') == False:
        TerminateWithError('The <t>time</i> variable does not have <i>units</i> attribute.')
    # elif hasattr(DatetimeObject,'calendar') == False:
        # TerminateWithError('The <t>time</i> variable does not have <i>calendar</i> attribute.')
    elif DatetimeObject.size < 2:
        TerminateWithError('The <i>time</i> variable size should be at least <tt>2</tt>.')

    # Longitude
    LongitudeNamesList = ['longitude','lon','long']
    LongitudeStandardNamesList = ['longitude']
    LongitudeObject,LongitudeName,LongitudeStandardName = SearchVariable(agg,LongitudeNamesList,LongitudeStandardNamesList)

    # Check longitude variable
    if LongitudeObject is None:
        TerminateWithError('Can not find the <i>longitude</i> variable in the netcdf file.')
    elif len(LongitudeObject.shape) != 1:
        TerminateWithError('The <t>longitude</i> variable dimension should be <tt>1<//t>.')
    elif LongitudeObject.size < 2:
        TerminateWithError('The <i>longitude</i> variable size should be at least <tt>2</tt>.')

    # Latitude
    LatitudeNamesList = ['latitude','lat']
    LatitudeStandardNamesList = ['latitude']
    LatitudeObject,LatitudeName,LatitudeStandardName = SearchVariable(agg,LatitudeNamesList,LatitudeStandardNamesList)

    # Check latitude variable
    if LatitudeObject is None:
        TerminateWithError('Can not find the <i>latitude</i> variable in the netcdf file.')
    elif len(LatitudeObject.shape) != 1:
        TerminateWithError('The <t>latitude</i> variable dimension should be <tt>1<//t>.')
    elif LatitudeObject.size < 2:
        TerminateWithError('The <i>latitude</i> variable size should be at least <tt>2</tt>.')

    return DatetimeObject,LongitudeObject,LatitudeObject

# =======================
# Load Velocity Variables
# =======================

def LoadVelocityVariables(agg):
    """
    Finds the following variables from the aggregation object agg.

    - Eastward velocity U
    - Northward velocity V
    """
    # East Velocity
    EastVelocityNamesList = ['east_vel','eastward_vel','u','east_velocity','eastward_velocity']
    EastVelocityStandardNamesList = ['surface_eastward_sea_water_velocity','eastward_sea_water_velocity']
    EastVelocityObject,EastVelocityName,EastVelocityStandardName = SearchVariable(agg,EastVelocityNamesList,EastVelocityStandardNamesList)

    # North Velocity
    NorthVelocityNamesList = ['north_vel','northward_vel','v','north_velocity','northward_velocity']
    NorthVelocityStandardNamesList = ['surface_northward_sea_water_velocity','northward_sea_water_velocity']
    NorthVelocityObject,NorthVelocityName,NorthVelocityStandardName = SearchVariable(agg,NorthVelocityNamesList,NorthVelocityStandardNamesList)

    return EastVelocityObject,NorthVelocityObject,EastVelocityName,NorthVelocityName,EastVelocityStandardName,NorthVelocityStandardName

# =================
# Prepare Datetimes
# =================

def PrepareDatetimes(DatetimeObject):
    """
    This is used in writer function.
    Converts date char format to datetime numeric format.
    This parses the times chars and converts them to date times.
    """

    # Datetimes units
    if (hasattr(DatetimeObject,'units')) and (DatetimeObject.units != ''):
        DatetimesUnit = DatetimeObject.units
    else:
        DatetimesUnit = 'days since 1970-01-01 00:00:00 UTC'

    # Datetimes calendar
    if (hasattr(DatetimeObject,'calendar')) and (DatetimeObject.calendar != ''):
        DatetimesCalendar = DatetimeObject.calendar
    else:
        DatetimesCalendar ='gregorian'

    # Datetimes
    DaysList = []
    OriginalDatetimes = DatetimeObject[:]

    if OriginalDatetimes.ndim == 1:

        # Datetimes in original dataset is already suitable to use
        Datetimes = OriginalDatetimes

    elif OriginalDatetimes.ndim == 2:

        # Datetime in original dataset is in the form of string. They should be converted to numerics
        for i in range(OriginalDatetimes.shape[0]):

            # Get row as string (often it is already a string, or a byte type)
            CharTime = numpy.chararray(OriginalDatetimes.shape[1])
            for j in range(OriginalDatetimes.shape[1]):
                CharTime[j] = OriginalDatetimes[i,j].astype('str')

            # Parse chars to integers
            Year = int(CharTime[0] + CharTime[1] + CharTime[2] + CharTime[3])
            Month = int(CharTime[5] + CharTime[6])
            Day = int(CharTime[8] + CharTime[9])
            Hour = int(CharTime[11] + CharTime[12])
            Minute = int(CharTime[14] + CharTime[15])
            Second = int(CharTime[17] + CharTime[18])

            # Create Day object
            DaysList.append(datetime.datetime(Year,Month,Day,Hour,Minute,Second))

        # Convert dates to numbers
        Datetimes = netCDF4.date2num(DaysList,units=DatetimesUnit,calendar=DatetimesCalendar)
    else:
        raise RuntimeError("Datetime ndim is more than 2.")

    return Datetimes,DatetimesUnit,DatetimesCalendar

# =============
# Get Time Info
# =============

def GetTimeInfo(DatetimeObject):
    """
    Get the initial time info and time duration.
    """

    Datetimes,DatetimesUnit,DatetimesCalendar = PrepareDatetimes(DatetimeObject)

    # Initial time
    InitialTime = Datetimes[0]
    InitialDatetimeObject = netCDF4.num2date(InitialTime,units=DatetimesUnit,calendar=DatetimesCalendar)

    InitialTimeDict = \
    {
        "Year": str(InitialDatetimeObject.year).zfill(4),
        "Month": str(InitialDatetimeObject.month).zfill(2),
        "Day": str(InitialDatetimeObject.day).zfill(2),
        "Hour": str(InitialDatetimeObject.hour).zfill(2),
        "Minute": str(InitialDatetimeObject.minute).zfill(2),
        "Second": str(InitialDatetimeObject.second).zfill(2),
        "Microsecond": str(InitialDatetimeObject.microsecond).zfill(6)
    }

    # Round off with microsecond
    if int(InitialTimeDict['Microsecond']) > 500000:
        InitialTimeDict['Microsecond'] = '000000'
        InitialTimeDict['Second'] = str(int(InitialTimeDict['Second'])+1)

    # Round off with second
    if int(InitialTimeDict['Second']) >= 60:
        ExcessSecond = int(InitialTimeDict['Second']) - 60
        InitialTimeDict['Second'] = '00'
        InitialTimeDict['Minute'] = str(int(InitialTimeDict['Minute'])+ ExcessSecond + 1)

    # Round off with minute
    if int(InitialTimeDict['Minute']) >= 60:
        ExcessMinute = int(InitialTimeDict['Minute']) - 60
        InitialTimeDict['Minute'] = '00'
        InitialTimeDict['Hour'] = str(int(InitialTimeDict['Hour']) + ExcessMinute + 1)

    # Round off with hour
    if int(InitialTimeDict['Hour']) >= 24:
        ExcessHour = int(InitialTimeDict['Hour']) - 24
        InitialTimeDict['Hour'] = '00'
        InitialTimeDict['Day'] = str(int(InitialTimeDict['Day']) + ExcessHour + 1)

    # Final time
    FinalTime = Datetimes[-1]
    FinalDatetimeObject = netCDF4.num2date(FinalTime,units=DatetimesUnit,calendar=DatetimesCalendar)

    FinalTimeDict = \
    {
        "Year": str(FinalDatetimeObject.year).zfill(4),
        "Month": str(FinalDatetimeObject.month).zfill(2),
        "Day": str(FinalDatetimeObject.day).zfill(2),
        "Hour": str(FinalDatetimeObject.hour).zfill(2),
        "Minute": str(FinalDatetimeObject.minute).zfill(2),
        "Second": str(FinalDatetimeObject.second).zfill(2),
        "Microsecond": str(FinalDatetimeObject.microsecond).zfill(6)
    }

    # Round off with microsecond
    if int(FinalTimeDict['Microsecond']) > 500000:
        FinalTimeDict['Microsecond'] = '000000'
        # FinalTimeDict['Second'] = str(int(InitialTimeDict['Second'])+1)  # Do not increase the second for final time

    # Round off with second
    if int(FinalTimeDict['Second']) >= 60:
        ExcessSecond = int(FinalTimeDict['Second']) - 60
        FinalTimeDict['Second'] = '00'
        FinalTimeDict['Minute'] = str(int(FinalTimeDict['Minute']) + ExcessSecond + 1)

    # Round off with minute
    if int(FinalTimeDict['Minute']) >= 60:
        ExcessMinute = int(FinalTimeDict['Minute']) - 60
        FinalTimeDict['Minute'] = '00'
        FinalTimeDict['Hour'] = str(int(FinalTimeDict['Hour']) + ExcessMinute+ 1)

    # Round off with hour
    if int(FinalTimeDict['Hour']) >= 24:
        ExcessHour = int(FinalTimeDict['Hour']) - 24
        FinalTimeDict['Hour'] = '00'
        FinalTimeDict['Day'] = str(int(FinalTimeDict['Day']) + ExcessHour + 1)

    # Find time unit
    DatetimesUnitString = DatetimesUnit[:DatetimesUnit.find('since')].replace(' ','')

    # Find time unit conversion to make times in unit of day
    if 'microsecond' in DatetimesUnitString:
        TimeUnitConversion = 1.0 / 1000000.0
    elif 'millisecond' in DatetimesUnitString:
        TimeUnitConversion = 1.0 / 1000.0
    elif 'second' in DatetimesUnitString:
        TimeUnitConversion = 1.0
    elif 'minute' in DatetimesUnitString:
        TimeUnitConversion = 60.0
    elif 'hour' in DatetimesUnitString:
        TimeUnitConversion = 3600.0
    elif 'day' in DatetimesUnitString:
        TimeUnitConversion = 24.0 * 3600.0

    # Time duration (in seconds)
    TimeDuration = numpy.fabs(Datetimes[-1] - Datetimes[0]) * TimeUnitConversion

    # Round off with microsecond
    # TimeDuration = numpy.floor(TimeDuration + 0.5)
    TimeDuration = numpy.floor(TimeDuration)

    # Day
    Residue = 0.0
    TimeDuration_Day = int(numpy.floor(TimeDuration/(24.0*3600.0)))
    Residue = TimeDuration - float(TimeDuration_Day) * (24.0*3600.0)

    # Hour
    TimeDuration_Hour = int(numpy.floor(Residue / 3600.0))
    Residue -= float(TimeDuration_Hour) * 3600.0

    # Minute
    TimeDuration_Minute = int(numpy.floor(Residue / 60.0))
    Residue -= float(TimeDuration_Minute) * 60.0

    # Second
    TimeDuration_Second = int(numpy.floor(Residue))

    TimeDurationDict = \
    {
        "Day": str(TimeDuration_Day),
        "Hour": str(TimeDuration_Hour).zfill(2),
        "Minute": str(TimeDuration_Minute).zfill(2),
        "Second": str(TimeDuration_Second).zfill(2)
    }

    # Datetime Size
    DatetimeSize = DatetimeObject.size

    # Create time info dictionary
    TimeInfo = \
    {
        "InitialTime": InitialTimeDict,
        "FinalTime": FinalTimeDict,
        "TimeDuration": TimeDurationDict,
        "TimeDurationInSeconds": str(TimeDuration),
        "DatetimeSize": str(DatetimeSize)
    }

    return TimeInfo

# ==============
# Get Space Info
# ==============

def GetSpaceInfo(LongitudeObject,LatitudeObject):
    """
    Get the dictionary of data bounds and camera bounds.
    """
    
    # Bounds for dataset
    MinLatitude = numpy.min(LatitudeObject[:])
    MaxLatitude = numpy.max(LatitudeObject[:])
    MinLongitude = numpy.min(LongitudeObject[:])
    MaxLongitude = numpy.max(LongitudeObject[:])

    # Cut longitude for overlapping longitudes
    if MaxLongitude - MinLongitude > 360:
        MaxLongitude = MinLongitude + 360

    # Center
    MidLongitude = 0.5 * (MinLongitude + MaxLongitude)
    MidLatitude = 0.5 * (MinLatitude + MaxLatitude)

    # Range (in meters)
    EarthRadius = 6378.1370e+3   # meters
    LatitudeRange = (MaxLatitude - MinLatitude) * (numpy.pi / 180.0) * EarthRadius
    LongitudeRange = (MaxLongitude - MinLongitude) * (numpy.pi / 180.0) * EarthRadius * numpy.cos(MidLatitude * numpy.pi / 180.0)

    # Resolutions
    LongitudeResolution = LongitudeObject.size
    LatitudeResolution = LatitudeObject.size

    # View Range
    ViewScale = 1.4  # This is used to show a bit larger area from left to right along LongitudeRange
    ViewRange = numpy.clip(LongitudeRange * ViewScale,0.0,2.0*EarthRadius * ViewScale)

    # Pitch Angle, measured from horizon downward. 45 degrees for small ranges, approaches 90 degrees for large ranges.
    PitchAngle = 45.0 + 45.0 * numpy.max([numpy.fabs(MaxLongitude-MinLongitude)/360.0,numpy.fabs(MaxLatitude-MinLatitude)/180.0])
    
    # Bounds for Camera
    LatitudeRatio = 0.2
    LatitudeSpan = MaxLatitude - MinLatitude

    # On very large latitude spans, the LatitudeRatgio becomes ineffective.
    CameraMinLatitude = numpy.clip(MinLatitude - LatitudeRatio * LatitudeSpan * (1.0 - LatitudeSpan/180.0),-90.0,90.0)
    CameraMaxLatitude = numpy.clip(MaxLatitude + LatitudeRatio * LatitudeSpan * (1.0 - LatitudeSpan/180.0),-90.0,90.0)

    LongitudeRatio = 0.2
    LongitudeSpan = MaxLongitude - MinLongitude

    # On very large longitude spans, the LongitudeRatio becomes ineffective.
    CameraMinLongitude = MidLongitude - numpy.clip(LongitudeSpan/2.0 + LongitudeRatio * LongitudeSpan * (1.0 - LongitudeSpan / 360.0),0.0,90.0)
    CameraMaxLongitude = MidLongitude + numpy.clip(LongitudeSpan/2.0 + LongitudeRatio * LongitudeSpan * (1.0 - LongitudeSpan / 360.0),0.0,90.0)

    DataBoundsDict = \
    {
        "MinLatitude": str(MinLatitude),
        "MidLatitude": str(MidLatitude),
        "MaxLatitude": str(MaxLatitude),
        "MinLongitude": str(MinLongitude),
        "MidLongitude": str(MidLongitude),
        "MaxLongitude": str(MaxLongitude)
    }

    DataRangeDict = \
    {
        "LongitudeRange": str(LongitudeRange),
        "LatitudeRange": str(LatitudeRange),
        "ViewRange": str(ViewRange),
        "PitchAngle": str(PitchAngle)
    }

    DataResolutionDict = \
    {
        "LongitudeResolution": str(LongitudeResolution),
        "LatitudeResolution": str(LatitudeResolution)
    }

    CameraBoundsDict = \
    {
        "MinLatitude": str(CameraMinLatitude),
        "MaxLatitude": str(CameraMaxLatitude),
        "MinLongitude": str(CameraMinLongitude),
        "MaxLongitude": str(CameraMaxLongitude),
    }

    # Space Info
    SpaceInfoDict = \
    {
        "DataResolution": DataResolutionDict,
        "DataBounds": DataBoundsDict,
        "DataRange": DataRangeDict,
        "CameraBounds": CameraBoundsDict
    }

    return SpaceInfoDict

# =================
# Get Velocity Name
# =================

def GetVelocityName(EastName,NorthName):
    """
    Given two names: EastName and NorthName, it finds the intersection of the names.
    Also it removes the redundand slashes, etc. 

    Exmaple:
    EastName:  surface_eastward_sea_water_velocity
    NorthName: surface_northward_sea_water_velocity

    Output VelocityName: surface_sea_water_velocity
    """

    # Split names based on a delimiter
    Delimiter = '_'
    EastNameSplitted = EastName.split(Delimiter)
    NorthNameSplitted = NorthName.split(Delimiter)

    VelocityNameList = []
    NumWords = numpy.min([len(EastNameSplitted),len(NorthNameSplitted)])
    for i in range(NumWords):
        if EastNameSplitted[i] == NorthNameSplitted[i]:
            VelocityNameList.append(EastNameSplitted[i])

    # Convert set to string
    if len(VelocityNameList) > 0:
        VelocityName = '_'.join(str(s) for s in VelocityNameList)
    else:
        VelocityName = ''

    return VelocityName

# =================
# Get Velocity Info
# =================

def GetVelocityInfo( \
        EastVelocityObject, \
        NorthVelocityObject, \
        EastVelocityName, \
        NorthVelocityName, \
        EastVelocityStandardName, \
        NorthVelocityStandardName):
    """
    Get dictionary of velocities.
    """

    # Get the number of indices to be selected for finding min and max.
    NumTimes = EastVelocityObject.shape[0]
    NumTimeIndices = 10 # Selecting 10 samples
    if NumTimeIndices > NumTimes:
        NumTimeIndices = NumTimes

    # The selection of random time indices to be used for finding min and max
    TimesIndices = numpy.random.randint(0,NumTimes-1,NumTimeIndices)

    # Min/Max velocities for each time frame
    EastVelocities_Mean = numpy.zeros(len(TimesIndices),dtype=float)
    EastVelocities_STD = numpy.zeros(len(TimesIndices),dtype=float)
    NorthVelocities_Mean = numpy.zeros(len(TimesIndices),dtype=float)
    NorthVelocities_STD = numpy.zeros(len(TimesIndices),dtype=float)

    # Find Min and Max of each time frame
    for k in range(len(TimesIndices)):

        TimeIndex = TimesIndices[k]

        with numpy.errstate(invalid='ignore'):
            EastVelocities_Mean[k]  = numpy.nanmean(EastVelocityObject[TimeIndex,:,:])
            EastVelocities_STD[k]   = numpy.nanstd(EastVelocityObject[TimeIndex,:,:])
            NorthVelocities_Mean[k] = numpy.nanmean(NorthVelocityObject[TimeIndex,:,:])
            NorthVelocities_STD[k]  = numpy.nanstd(NorthVelocityObject[TimeIndex,:,:])

    # Mean and STD of Velocities among all time frames
    EastVelocity_Mean  = numpy.nanmean(EastVelocities_Mean)
    EastVelocity_STD   = numpy.nanmean(EastVelocities_STD)
    NorthVelocity_Mean = numpy.nanmean(NorthVelocities_Mean)
    NorthVelocity_STD  = numpy.nanmean(NorthVelocities_STD)

    # Min/Max of Velocities, assuming u and v have Gaussian distributions
    Scale = 4.0
    MinEastVelocity  = EastVelocity_Mean  - Scale * EastVelocity_STD
    MaxEastVelocity  = EastVelocity_Mean  + Scale * EastVelocity_STD
    MinNorthVelocity = NorthVelocity_Mean - Scale * NorthVelocity_STD
    MaxNorthVelocity = NorthVelocity_Mean + Scale * NorthVelocity_STD

    # An estimate for max velocity speed. If u and v has Gaussian distributions, the velocity speed has Chi distribution
    Velocity_Mean = numpy.sqrt(EastVelocity_Mean**2 + EastVelocity_STD**2 + NorthVelocity_Mean**2 + NorthVelocity_STD**2)
    TypicalVelocitySpeed = 4.0 * Velocity_Mean

    # Get the velocity name from east and north names
    if (EastVelocityStandardName != '') and (NorthVelocityStandardName != ''):
        VelocityStandardName = GetVelocityName(EastVelocityStandardName,NorthVelocityStandardName)
    else:
        VelocityStandardName = ''

    # Create a Velocity Info Dict
    VelocityInfoDict = \
    {
        "EastVelocityName": EastVelocityName,
        "NorthVelocityName": NorthVelocityName,
        "EastVelocityStandardName": EastVelocityStandardName,
        "NorthVelocityStandardName": NorthVelocityStandardName,
        "VelocityStandardName": VelocityStandardName,
        "MinEastVelocity": str(MinEastVelocity),
        "MaxEastVelocity": str(MaxEastVelocity),
        "MinNorthVelocity": str(MinNorthVelocity),
        "MaxNorthVelocity": str(MaxNorthVelocity),
        "TypicalVelocitySpeed": str(TypicalVelocitySpeed)
    }

    return VelocityInfoDict

# ====
# Main
# ====

def main(argv):
    """
    Reads a netcdf file and returns data info.

    Notes:

        - If the option -V is used to scan min and max of velocities,
          we do not find the min and max of velocity for all time frames. This is
          becasue if the nc file is large, it takes a long time. Also we do not load the whole
          velocities like U[:] or V[:] becase it fthe data is large, the netCDF4 package raises
          an error.
    """

    # Fill output with defaults
    DatasetInfoDict = \
    {
        "Scan": \
        {
            "ScanStatus": True,
            "Message": ""
        },
        "TimeInfo": None,
        "SpaceInfo": None,
        "VelocityInfo": None
    }

    # Parse arguments
    InputFilename,ScanVelocityStatus = ParseArguments(argv)

    # Open file
    agg = LoadDataset(InputFilename)

    # Load variables
    DatetimeObject,LongitudeObject,LatitudeObject = LoadTimeAndSpaceVariables(agg)

    # Get Time Info
    TimeInfoDict = GetTimeInfo(DatetimeObject)
    DatasetInfoDict['TimeInfo'] = TimeInfoDict

    # Get Space Info
    SpaceInfoDict = GetSpaceInfo(LongitudeObject,LatitudeObject)
    DatasetInfoDict['SpaceInfo'] = SpaceInfoDict

    # Velocities
    if ScanVelocityStatus == True:

        # Get velocity objects
        EastVelocityObject,NorthVelocityObject,EastVelocityName,NorthVelocityName,EastVelocityStandardName,NorthVelocityStandardName = LoadVelocityVariables(agg)
        VelocityInfoDict = GetVelocityInfo(EastVelocityObject,NorthVelocityObject,EastVelocityName,NorthVelocityName,EastVelocityStandardName,NorthVelocityStandardName)
        DatasetInfoDict['VelocityInfo'] = VelocityInfoDict

    agg.close()

    DatasetInfoJson = json.dumps(DatasetInfoDict,indent=4)
    print(DatasetInfoJson)
    sys.stdout.flush()

# ===========
# System Main
# ===========

if __name__ == "__main__":

    # Converting all warnings to error
    # warnings.simplefilter('error',UserWarning)
    warnings.filterwarnings("ignore",category=numpy.VisibleDeprecationWarning)
    warnings.filterwarnings("ignore",category=DeprecationWarning)

    # Main function
    main(sys.argv)
