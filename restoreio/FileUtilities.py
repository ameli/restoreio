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

import re
import sys
import os
try:
    # Python 3
    from urllib.parse import urlparse
except ImportError:
    # python 2
    from urlparse import urlparse

# =====================================
# Get String Components Of File Address
# =====================================

def GetStringComponentsOfFileAddress(FullPathInputFilename):
    """
    The FullPathInputFilename can be a local file address or a URL.
    This function finds the filename after the slash in URL, or after a directory separator

    Example of input:
        http://aa.bb.com/cc/filename001AA.nc, 

    Example of outputs:
        BaseDirOrURL: http://aa.bb.com/cc
        BaseFilename: filename001AA
        FileExtention: nc
    """

    # Check input
    if len(FullPathInputFilename) < 1:
        print('ERROR: Local or remote Filename is not provided.')
        sys.exit(1)

    elif FullPathInputFilename.rfind(' ') != -1:
        print('ERROR: Filename or URL can not contain a white space.')
        sys.exit(1)

    # Get the last part of string after the slash
    LastSlashPosition = FullPathInputFilename.rfind('/')

    # Detect whether the address is local or remote
    if bool(urlparse(FullPathInputFilename).netloc):

        # The address is a URL
        if LastSlashPosition < 1:
            print('ERROR: Can not find the filename from the remote address. The URL does not seem to have a slash <tt>/</tt> separator.')
            sys.exit(1)
        elif LastSlashPosition == len(FullPathInputFilename) - 1:
            print('ERROR: Can not find filename from the given URL. The URL should not end with slash <tt>/</tt>.')
            sys.exit(1)

        # Separate URL or Directory from the filename
        BaseDirOrURL = FullPathInputFilename[:LastSlashPosition]
        Filename = FullPathInputFilename[LastSlashPosition+1:]
        if Filename == "":
            print('ERROR: Can not find the filename from the given URL. The filename seems to have zero length.')
            sys.exit(1)

    else:

        # The address is a local file
        if LastSlashPosition < 0:
            BaseDirOrURL = '.'
            Filename = FullPathInputFilename
        else:
            BaseDirOrURL = FullPathInputFilename[:LastSlashPosition]
            Filename = FullPathInputFilename[LastSlashPosition+1:]

    # Check filename
    if len(Filename) < 1:
        print('ERROR: Filename in the remote or local address has zero length.')
        sys.exit(1)

    # Get the Base filename
    LastDotPosition = Filename.rfind('.')
    if LastDotPosition < 0:
        print('ERROR: The filename does not seem to have a file extension.')
        sys.exit(1)

    # base filename and URL
    BaseFilename = Filename[:LastDotPosition]
    FileExtension = Filename[LastDotPosition+1:]

    # Check BaseFilename
    if len(BaseFilename) < 1:
        print('ERROR: The base filename has zero length.')
        sys.exit(1)


    return BaseDirOrURL, BaseFilename, FileExtension

# ================================
# Get BaseFilename Iterator String
# ================================

def GetBaseFilenameIteratorString(BaseFilename):
    """
    Given the baseFilename like MyFile001AA.nc, this returns 001 as string.
    """

    # Detect numeric iterators in the BaseFilename
    NumbersList = re.findall("\d+", BaseFilename)

    if len(NumbersList) < 1:
        print("ERROR: Can not find numeric iterators in the filename.")
        sys.exit(1)

    BaseFilenameIteratorString = NumbersList[-1]

    return BaseFilenameIteratorString

# =================================
# Generate List Of Iterators String
# =================================

def GenerateListOfIteratorsString( \
        BaseFilenameIteratorString, \
        MinIteratorString, \
        MaxIteratorString):
    """
    Generates the list of iterators between min and max. The leading zeros are preserved.
    """

    IteratorsStringList = []

    # Conversion to integers
    MinIterator = int(MinIteratorString)
    MaxIterator = int(MaxIteratorString)

    # Check order
    if MinIterator > MaxIterator:
        print('ERROR: When using multiple files to process, the <i>Mininum file iterator</i> can not be larger than <i>Maximum file iterator</i>.')
        sys.exit(1)
    elif len(MinIteratorString) > len(MaxIteratorString):
        print('ERROR: When using multiple files to process, the <b>length</b> of <i>minimum file iterator</i> string (including leading zeros) can not be more than the length of <i>maximum file iterator</i> string.')
        sys.exit(1)

    # Preserver the minimum of string length
    MinStringLength = max(len(MinIteratorString), len(BaseFilenameIteratorString))

    # Iterate
    for Iterator in range(MinIterator, MaxIterator+1):
        IteratorString = str(Iterator).zfill(MinStringLength)
        IteratorsStringList.append(IteratorString)

    return IteratorsStringList

# ==================================
# Get Full Path Input Filenames List
# ==================================

def GetFullPathInputFilenamesList( \
        FullPathInputFilename, \
        ProcessMultipleFiles, \
        MinIteratorString, \
        MaxIteratorString):
    """
    Get filenames of local or remote datasets.
    The data can be a single file or multiple separate dataset in separate files.
    """

    # Get the dataset FullPathInputFilename, which might be a url or a address on local machine.

    # Check FullPathInputFilename
    if FullPathInputFilename == '':
        print('ERROR: Input data URL is empty.')
        sys.exit(1)

    # Get the configurations
    GivenFullPathInputFilename = FullPathInputFilename + ''
    
    # Initiate the list of files
    FullPathInputFilenamesList = []
    InputBaseFilenamesList = []

    # Check if files are single or multiple
    if ProcessMultipleFiles == False:

        # Append the single fullpath filename
        FullPathInputFilenamesList.append(GivenFullPathInputFilename)

        # Append the single base filename
        BaseFilenameAndExtension = os.path.basename(FullPathInputFilename)
        LastDotPosition = BaseFilenameAndExtension.rfind('.')
        BaseFilename = BaseFilenameAndExtension[:LastDotPosition]
        InputBaseFilenamesList.append(BaseFilename)

    else:

        # Process multiple files

        # Get numeric Iterator in the filename string
        BaseDirOrURL, BaseFilename, FileExtension = GetStringComponentsOfFileAddress(GivenFullPathInputFilename)

        # Find the iterator in the basefilename
        BaseFilenameIteratorString = GetBaseFilenameIteratorString(BaseFilename)

        # All range of Iterators as string
        IteratorsStringList = GenerateListOfIteratorsString(BaseFilenameIteratorString, MinIteratorString, MaxIteratorString)

        for i in range(len(IteratorsStringList)):

            # Full Path Filename. Replace the last occurance of the iterator string
            FullPathFilenameWithNewIterator = GivenFullPathInputFilename[::-1].replace(BaseFilenameIteratorString[::-1], IteratorsStringList[i][::-1], 1)[::-1]
            FullPathInputFilenamesList.append(FullPathFilenameWithNewIterator)

            # Base Filename
            BaseFilenameWithNewIterator = BaseFilename[::-1].replace(BaseFilenameIteratorString[::-1], IteratorsStringList[i][::-1], 1)[::-1]
            InputBaseFilenamesList.append(BaseFilenameWithNewIterator)

    return FullPathInputFilenamesList, InputBaseFilenamesList

# ===================================
# Get Full Path Output Filenames List
# ===================================

def GetFullPathOutputFilenamesList( \
        FullPathOutputFilename, \
        ProcessMultipleFiles, \
        MinIteratorString, \
        MaxIteratorString):
    """
    This function creates a list of output files.
    If ProcessMultipleFiles is false, then the output fles list contans only one file, and it is the FullPathOutputFilename that is given already.
    If ProcessMultipleFiles is true, then it decomposes the FullPathOutputFilename to OutputFilePath + '/' + OutputBaseFilename + '.nc', and then
    generates a list with:
        OutputFilePath + '/' + OutputBaseFilename + '-IteratorNumber' + '.nc'
    where the iterator number runs trom MinIteratorString to maxIteratorString.
    """

    FullPathOutputFilenamesList = []

    if ProcessMultipleFiles == False:

        # Do not produce multiple files.
        FullPathOutputFilenamesList.append(FullPathOutputFilename)

    else:

        # Produce multiple files
        OutputFilePath, OutputBaseFilename, OutputFileExtension = GetStringComponentsOfFileAddress(FullPathOutputFilename)
        IteratorStringList = GenerateListOfIteratorsString(MinIteratorString, MinIteratorString, MaxIteratorString)

        for IteratorString in IteratorStringList:
            IteratedFullPathOutputFilename = OutputFilePath + '/' + OutputBaseFilename + '-' + IteratorString + '.' + OutputFileExtension
            FullPathOutputFilenamesList.append(IteratedFullPathOutputFilename)

    return FullPathOutputFilenamesList

# ======================
# Archive Multiple Files
# ======================

def ArchiveMultipleFiles( \
        FullPathOutputFilename, \
        FullPathOutputFilenamesList, \
        BaseFilenamesListInZipFile):
    """
    Archives multiple output files, and then deletes the original files to clean the directory.

    Inputs:

        - ZipBaseFilename: The base filename of the zip file.
        - ZipFilePath: The file path of the zip file.
        - FullPathOutputFilenamesList: The lisft of files that are going to be zipped. These files will be deleted after they were coppied to the zip file.
        - BaseFilenamesListInZipFile: The name of each of the FullPathOutputFilenamesList when they are coppied into the zip archive.
    """

    import zipfile

    print('Message: Zip output files ...')

    # Replace the output file extension to zip for the name of the zipfile
    LastDotPosition = FullPathOutputFilename.rfind('.')
    FullPathBaseFilename = FullPathOutputFilename[:LastDotPosition]

    # Zip files (We store zip files where the CZML files can be stored on server)
    ZipFullPathFilename = FullPathBaseFilename + '.zip'
    ZipFileObject = zipfile.ZipFile(ZipFullPathFilename, "w")

    for i in range(len(FullPathOutputFilenamesList)):

        OutputFilenameInZipFile = BaseFilenamesListInZipFile[i] + ".nc"
        ZipFileObject.write(FullPathOutputFilenamesList[i], arcname=OutputFilenameInZipFile)

    ZipFileObject.close()


    # Delete Files and only keep the zip file
    print('Message: Clean directory ...')
    for i in range(len(FullPathOutputFilenamesList)):
        os.remove(FullPathOutputFilenamesList[i])
