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
import re

__all__ = ['WrappedNewlineFormatter']


# =====================================
# Description Wrapped Newline Formatter
# =====================================

class DescriptionWrappedNewlineFormatter(argparse.HelpFormatter):
    """
    An argparse formatter for formatting only the "prelog" and "epilog" parts
    of the help message. That is, the message that comes before and after
    the description of options.

    To format the "required arguments" and "optional argument", see
    "WrappedNewlineFormatter".

    This formatter does the followings:

        1. Wraps lines to the screen width.
        2. Connects all lines into one line, provide that the lines
           are separated by only one "\n", not two, "\n\n" (a paragraph
           separation)
        3. Keeps a two successive new lines (a paragraph separation) as is.
        4. Those lines that start with "$" (code lines) will not be wrapped,
           instead, they will be indented.
        5. In the code line, the word "%s" is replaced by the program name.

    For example, by assuming a text wrap of length 40, the following text:

    '''
       A line.
       Another new line, but in the same paragraph.

       A new paragraph.
       Second line of paragraph.

       $ %s options ...
    '''

    is converted to the following (in this example, wrap length is 40)

    '''
    A line. Another new line, but in the
    same paragraph.

    A new paragraph. Second line of
    paragraph.

       $ program_name options ...
    '''
    """

    def _fill_text(self, text, width, indent):

        # # Strip the indent from the original python definition
        text = textwrap.dedent(text)
        text = textwrap.indent(text, indent)  # Apply any requested indent.

        # Convert new lines (\n) to space " ", not not double new lines (\n\n)
        text = re.sub("(?<!(\\n))(\\n)(?!(\\n))", " ", text)

        # Remove while space in the beginning
        text = re.sub(r"^\s+", "", text)

        # Remove more than one white space
        text = re.sub(" +", " ", text)

        # Limit the length of wrap to max 80 columns.
        # max_width = 80
        # if width > max_width: width = max_width

        # Wrap
        wrapper = textwrap.TextWrapper(width=width, replace_whitespace=False)

        lines = text.splitlines()
        wrapped_lines = []

        for line in lines:

            # Remove white space in the beginning of a line
            line = re.sub(r"^\s+", "", line)

            # Remove two or more white space
            line = re.sub(" +", " ", line)

            if line.strip() == "":
                # Just add an empty new line
                wrapped_lines += [""]

            elif line[0] == "$":
                #  Do not wrap code lines, rather, add indent
                line = r"   " + line % (self._prog)
                wrapped_lines += [line]

            else:
                # Separate lines that begin with enumeration
                splitted_line = re.split(r'(?m)^(\d+)\.\s+', line)

                if len(splitted_line) > 1:

                    # Line begins with enumeration.
                    num = splitted_line[1]
                    line = splitted_line[2]

                    indent_length = len(num) + 2
                    initial_indent = ' '*indent_length

                    # Wrap
                    temp_wrapped_lines = textwrap.wrap(
                            line, width=width-indent_length)

                    for i in range(len(temp_wrapped_lines)):

                        if i == 0:
                            temp_wrapped_lines[i] = num + '. ' + \
                                    temp_wrapped_lines[i]
                        else:
                            temp_wrapped_lines[i] = initial_indent + \
                                    temp_wrapped_lines[i]

                    wrapped_lines += temp_wrapped_lines

                else:
                    # Line does not begin with enumeration. Wrap as usual
                    wrapped_lines += wrapper.wrap(line)

        wrapped_text = '\n'.join(wrapped_lines)

        return wrapped_text


# =========================
# Wrapped Newline Formatter
# =========================

class WrappedNewlineFormatter(DescriptionWrappedNewlineFormatter):
    """
    An argparse formatter for formatting only the "required arguments" and
    "optional arguments" parts of the help message. That is, the message that
    comes in between the relog and epilog.

    To format the "prelog" and "epilog", see
    "DescriptionWrappedNewlineFormatter".

    This formatter does the followings:

        1. Wraps lines to the screen width.
        2. Connects all lines into one line, provide that the lines
           are separated by only one "\n", not two, "\n\n" (a paragraph
           separation)
        3. Changes a two successive new lines (a paragraph separation) to one
           new line.

    For example, by assuming a text wrap of length 40, the following text:

    '''
       1. A line.
       Another new line, but in the same paragraph.

       A new paragraph.
       Second line of paragraph.

       $ code
    '''

    is converted to the following (in this example, wrap length is 40)

    '''
    A line. Another new line, but in the
    same paragraph.
    A new paragraph. Second line of
    paragraph.
    $ code
    '''
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
        wrapper = textwrap.TextWrapper(width=width, replace_whitespace=False)

        lines = text.splitlines()
        wrapped_lines = []

        for line in lines:

            # Remove white space in the beginning of a line
            line = re.sub(r"^\s+", "", line)

            # Remove two or more white space
            line = re.sub(" +", " ", line)

            if line.strip() == "":
                # Do not add a empty new line.
                continue

            else:
                # Wrap line
                wrapped_lines += wrapper.wrap(line)

        wrapped_lines.append('')

        return wrapped_lines
