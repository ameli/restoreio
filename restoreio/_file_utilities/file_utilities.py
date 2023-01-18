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

__all__ = ['get_fullpath_input_filenames_list',
           'get_fullpath_output_filenames_list', 'archive_multiple_files']


# =====================================
# Get String Components Of File Address
# =====================================

def _get_string_components_of_file_address(fullpath_input_filename):
    """
    The fullpath_input_filename can be a local file address or a URL.
    This function finds the filename after the slash in URL, or after a
    directory separator

    Example of input:
        http://aa.bb.com/cc/filename001AA.nc,

    Example of outputs:
        base_dir_or_url: http://aa.bb.com/cc
        base_filename: filename001AA
        file_extension: nc
    """

    # Check input
    if len(fullpath_input_filename) < 1:
        print('ERROR: Local or remote filename is not provided.')
        sys.exit(1)

    elif fullpath_input_filename.rfind(' ') != -1:
        print('ERROR: filename or URL can not contain a white space.')
        sys.exit(1)

    # Get the last part of string after the slash
    last_slash_position = fullpath_input_filename.rfind('/')

    # Detect whether the address is local or remote
    if bool(urlparse(fullpath_input_filename).netloc):

        # The address is a URL
        if last_slash_position < 1:
            print('ERROR: Can not find the filename from the remote ' +
                  'address. The URL does not seem to have a slash ' +
                  '<tt>/</tt> separator.')
            sys.exit(1)
        elif last_slash_position == len(fullpath_input_filename) - 1:
            print('ERROR: Can not find filename from the given URL. The ' +
                  'URL should not end with slash <tt>/</tt>.')
            sys.exit(1)

        # Separate URL or Directory from the filename
        base_dir_or_url = fullpath_input_filename[:last_slash_position]
        filename = fullpath_input_filename[last_slash_position+1:]
        if filename == "":
            print('ERROR: Can not find the filename from the given URL. ' +
                  'The filename seems to have zero length.')
            sys.exit(1)

    else:

        # The address is a local file
        if last_slash_position < 0:
            base_dir_or_url = '.'
            filename = fullpath_input_filename
        else:
            base_dir_or_url = fullpath_input_filename[:last_slash_position]
            filename = fullpath_input_filename[last_slash_position+1:]

    # Check filename
    if len(filename) < 1:
        print('ERROR: filename in the remote or local address has zero ' +
              'length.')
        sys.exit(1)

    # Get the Base filename
    last_dot_position = filename.rfind('.')
    if last_dot_position < 0:
        print('ERROR: The filename does not seem to have a file extension.')
        sys.exit(1)

    # base filename and URL
    base_filename = filename[:last_dot_position]
    file_extension = filename[last_dot_position+1:]

    # Check base_filename
    if len(base_filename) < 1:
        print('ERROR: The base filename has zero length.')
        sys.exit(1)

    return base_dir_or_url, base_filename, file_extension


# =================================
# Get base filename Iterator String
# =================================

def _get_base_filename_iterator_string(base_filename):
    """
    Given the base filename like MyFile001AA.nc, this returns 001 as string.
    """

    # Detect numeric iterators in the base_filename
    numbers_list = re.findall(r"\d+", base_filename)

    if len(numbers_list) < 1:
        print("ERROR: Can not find numeric iterators in the filename.")
        sys.exit(1)

    base_filename_iterator_string = numbers_list[-1]

    return base_filename_iterator_string


# =================================
# Generate List Of Iterators String
# =================================

def _generate_list_of_iterators_string(
        base_filename_iterator_string,
        min_iterator_string,
        max_iterator_string):
    """
    Generates the list of iterators between min and max. The leading zeros are
    preserved.
    """

    iterators_string_list = []

    # Conversion to integers
    min_iterator = int(min_iterator_string)
    max_iterator = int(max_iterator_string)

    # Check order
    if min_iterator > max_iterator:
        print('ERROR: When using multiple files to process, the <i>Minimum ' +
              'file iterator</i> can not be larger than <i>Maximum file ' +
              'iterator</i>.')
        sys.exit(1)
    elif len(min_iterator_string) > len(max_iterator_string):
        print('ERROR: When using multiple files to process, the <b>length' +
              '</b> of <i>minimum file iterator</i> string (including ' +
              'leading zeros) can not be more than the length of <i>' +
              'maximum file iterator</i> string.')
        sys.exit(1)

    # Preserver the minimum of string length
    min_string_length = max(len(min_iterator_string),
                            len(base_filename_iterator_string))

    # Iterate
    for iterator in range(min_iterator, max_iterator+1):
        iterator_string = str(iterator).zfill(min_string_length)
        iterators_string_list.append(iterator_string)

    return iterators_string_list


# ==================================
# Get Full Path Input filenames List
# ==================================

def get_fullpath_input_filenames_list(
        fullpath_input_filename,
        min_iterator_string,
        max_iterator_string):
    """
    Get filenames of local or remote datasets. The data can be a single file or
    multiple separate dataset in separate files.
    """

    # Get the dataset fullpath_input_filename, which might be a url or a
    # address on local machine.

    # Check fullpath_input_filename
    if fullpath_input_filename == '':
        print('ERROR: Input data URL is empty.')
        sys.exit(1)

    # Get the configurations
    given_fullpath_input_filename = fullpath_input_filename + ''

    # Initiate the list of files
    fullpath_input_filenames_list = []
    input_base_filenames_list = []

    # Check if files are single or multiple
    if (min_iterator_string == '') and (max_iterator_string == ''):

        # Append the single fullpath filename
        fullpath_input_filenames_list.append(given_fullpath_input_filename)

        # Append the single base filename
        base_filename_and_extension = os.path.basename(fullpath_input_filename)
        last_dot_position = base_filename_and_extension.rfind('.')
        base_filename = base_filename_and_extension[:last_dot_position]
        input_base_filenames_list.append(base_filename)

    else:

        # Process multiple files

        # Get numeric Iterator in the filename string
        base_dir_or_url, base_filename, file_extension = \
                _get_string_components_of_file_address(
                        given_fullpath_input_filename)

        # Find the iterator in the base filename
        base_filename_iterator_string = _get_base_filename_iterator_string(
                base_filename)

        # All range of Iterators as string
        iterators_string_list = _generate_list_of_iterators_string(
                base_filename_iterator_string,
                min_iterator_string,
                max_iterator_string)

        for i in range(len(iterators_string_list)):

            # Full Path filename. Replace the last occurrence of the iterator
            # string
            fullpath_filename_with_new_iterator = \
                    given_fullpath_input_filename[::-1].replace(
                            base_filename_iterator_string[::-1],
                            iterators_string_list[i][::-1], 1)[::-1]
            fullpath_input_filenames_list.append(
                    fullpath_filename_with_new_iterator)

            # Base filename
            base_filename_with_new_iterator = base_filename[::-1].replace(
                    base_filename_iterator_string[::-1],
                    iterators_string_list[i][::-1], 1)[::-1]
            input_base_filenames_list.append(base_filename_with_new_iterator)

    return fullpath_input_filenames_list, input_base_filenames_list


# ===================================
# Get Full Path Output filenames List
# ===================================

def get_fullpath_output_filenames_list(
        fullpath_output_filename,
        min_iterator_string,
        max_iterator_string):
    """
    This function creates a list of output files.
    If min and max file indices are empty, then the output files list contains
    only one file, and it is the fullpath_output_filename that is given
    already.

    If min and max file indies are not empty, then it decomposes the
    fullpath_output_filename to

        output_file_path + '/' + output_base_filename + '.nc'

    and then generates a list with:

        output_file_path + '/' + output_base_filename + '-iterator_num' + '.nc'

    where the iterator number runs from min_iterator_string to
    max_iterator_string.
    """

    fullpath_output_filenames_list = []

    if (min_iterator_string == '') and (max_iterator_string == ''):

        # Do not produce multiple files.
        fullpath_output_filenames_list.append(fullpath_output_filename)

    else:

        # Produce multiple files
        output_file_path, output_base_filename, output_file_extension = \
                _get_string_components_of_file_address(
                        fullpath_output_filename)
        iterator_string_list = _generate_list_of_iterators_string(
                min_iterator_string, min_iterator_string, max_iterator_string)

        for iterator_string in iterator_string_list:
            iterated_fullpath_output_filename = output_file_path + '/' + \
                    output_base_filename + '-' + iterator_string + '.' + \
                    output_file_extension
            fullpath_output_filenames_list.append(
                    iterated_fullpath_output_filename)

    return fullpath_output_filenames_list


# ======================
# Archive Multiple Files
# ======================

def archive_multiple_files(
        fullpath_output_filename,
        fullpath_output_filenames_list,
        base_filenames_list_in_zip_file):
    """
    Archives multiple output files, and then deletes the original files to
    clean the directory.

    Inputs:

        - Zipbase_filename: The base filename of the zip file.
        - ZipFilePath: The file path of the zip file.
        - fullpath_output_filenames_list: The list of files that are going to
          be zipped. These files will be deleted after they were copied to the
          zip file.
        - base_filenames_list_in_zip_file: The name of each of the
          fullpath_output_filenames_list when they are copied into the zip
          archive.
    """

    import zipfile

    print('Message: Zip output files ...')

    # Replace the output file extension to zip for the name of the zipfile
    last_dot_position = fullpath_output_filename.rfind('.')
    fullpath_base_filename = fullpath_output_filename[:last_dot_position]

    # Zip files (We store zip files where CZML files can be stored on server)
    zip_fullpath_filename = fullpath_base_filename + '.zip'
    zip_file_obj = zipfile.ZipFile(zip_fullpath_filename, "w")

    for i in range(len(fullpath_output_filenames_list)):

        output_filename_in_zip_file = base_filenames_list_in_zip_file[i] + \
                ".nc"
        zip_file_obj.write(fullpath_output_filenames_list[i],
                           arcname=output_filename_in_zip_file)

    zip_file_obj.close()

    # Delete Files and only keep the zip file
    print('Message: Clean directory ...')
    for i in range(len(fullpath_output_filenames_list)):
        os.remove(fullpath_output_filenames_list[i])
