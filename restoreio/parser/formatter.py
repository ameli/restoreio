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


import argparse
import textwrap
import itertools
import re


# =====================================
# Description Wrapped Newline Formatter
# =====================================

class DescriptionWrappedNewlineFormatter(argparse.HelpFormatter):
    """
    An argparse formatter that:

    * preserves newlines (like argparse.RawDescriptionHelpFormatter),
    * removes leading indent (great for multiline strings),
    * and applies reasonable text wrapping.
    """

    def _fill_text(self, text, width, indent):
        # # Strip the indent from the original python definition
        text = textwrap.dedent(text)
        text = textwrap.indent(text, indent)  # Apply any requested indent.
        # text = text.splitlines()  # Make a list of lines
        # text = [textwrap.fill(line, width) for line in text]  # Wrap each line
        # text = "\n".join(text)  # Join the lines again
        #
        # return text

        # Allow multiline strings to have common leading indentation.
        # text = textwrap.dedent(text)

        # Convert new lines (\n) to space " ", not not double new lines (\n\n)
        text = re.sub("(?<!(\\n))(\\n)(?!(\\n))", " ", text)

        # Remove more than one white space
        # text = re.sub(" +", " ", text)

        # Remove while space in the beginning
        text = re.sub(r"^\s+", "", text)

        # Limit the length of wrap to max 80 columns.
        # max_width = 80
        # if width > max_width: width = max_width

        # Wrap
        wrapper = textwrap.TextWrapper(width = width)

        # Make list of lines
        lines_list = [wrapper.wrap(i) for i in text.split('\n') if i != '']
        lines_list = list(itertools.chain.from_iterable(lines_list))

        # Add one additional line after each option
        lines_list.append('')

        text = "\n".join(lines_list)  # Join the lines again

        return text


class WrappedNewlineFormatter(DescriptionWrappedNewlineFormatter):
    """
    An argparse formatter that:

    * preserves newlines (like argparse.RawTextHelpFormatter),
    * removes leading indent and applies reasonable text wrapping.
    * applies to all help text (description, arguments, epilogue).
    """
    def _split_lines(self, text, width):

        # Allow multiline strings to have common leading indentation.
        text = textwrap.dedent(text)

        # Convert new lines (\n) to space " ", not not double new lines (\n\n)
        text = re.sub("(?<!(\\n))(\\n)(?!(\\n))", " ", text)

        # Remove more than one white space
        text = re.sub(" +", " ", text)

        # Remove while space in the beginning
        text = re.sub(r"^\s+", "", text)

        # Limit the length of wrap to max 80 columns.
        # max_width = 80
        # if width > max_width: width = max_width

        # Wrap
        wrapper = textwrap.TextWrapper(width = width)

        # Make list of lines
        lines_list = [wrapper.wrap(i) for i in text.split('\n') if i != '']
        lines_list = list(itertools.chain.from_iterable(lines_list))

        # Add one additional line after each option
        lines_list.append('')

        return lines_list
