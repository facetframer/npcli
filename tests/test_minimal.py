# make code as python 3 compatible as possible
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import io
import os
import tempfile
import unittest

from npcli.npcli import run

class NpcliTest(unittest.TestCase):
    def setUp(self):
        self.direc = tempfile.mkdtemp()

    def run_cli(self, stdin, *args):
        result = run(stdin, args)
        return b''.join(result)

    def test_basic(self):
        self.assertEquals(self.run_cli(io.BytesIO(b'1\n2\n3\n'), 'd.sum()'), b'6.0\n')

    def test_multiple_streams(self):
        source2 = os.path.join(self.direc, 'source2')
        source3 = os.path.join(self.direc, 'source3')
        with io.open(source2, 'w', encoding='utf8') as stream:
            stream.write(u'8\n16\n32\n')
        with io.open(source3, 'w', encoding='utf8') as stream:
            stream.write(u'64\n128\n256\n')
        self.assertEquals(self.run_cli(io.BytesIO(b'1\n2\n4\n'), 'd + d1 + d2', source2, source3), b'73.0\n146.0\n292.0\n')

    def test_complex_expressions(self):
        self.run_cli(io.BytesIO(b'1\n2\n3\n'), '--repr',  'd[1]')
        self.run_cli(io.BytesIO(b'1\n2\n3\n'), '--repr',  'd[1:]')
        self.run_cli(io.BytesIO(b'1 11\n2 22\n3 33\n'), '--repr',  'd[0,0]')
        self.run_cli(io.BytesIO(b'1 11\n2 22\n3 33\n'), '--repr',  'd[::-1,::1]')

        self.run_cli(io.BytesIO(b'1\n2\n3\n'), '--repr',  '(d,)')
        self.run_cli(io.BytesIO(b'1\n2\n3\n'), '--repr',  '[d]')
        self.run_cli(io.BytesIO(b'1\n2\n3\n'), '--repr',  '[x+1 for x in d]')
        self.run_cli(io.BytesIO(b'1\n2\n3\n'), '--repr',  'd + 1')
        self.run_cli(io.BytesIO(b'1\n2\n3\n'), '--repr', '--', '-d')
        self.run_cli(io.BytesIO(b'1\n2\n3\n'), '--repr',  '"hello"')
        self.run_cli(io.BytesIO(b'1\n2\n3\n'), '--repr',  'd < 2')
        self.run_cli(io.BytesIO(b'1\n2\n3\n'), '--repr',  'd ** 2')
        self.run_cli(io.BytesIO(b'1\n2\n3\n'), '--repr',  'd.sum()')

    def test_assignment(self):
        result = self.run_cli(io.BytesIO(b'1\n'),  'a = d', '-e', 'a')
        self.assertEquals(result, b'1.0\n')

    def test_modules(self):
        result = self.run_cli(io.BytesIO(b'-1\n'), '-m', 'numpy', 'numpy.abs(d)')
        self.assertEquals(result, b'1.0\n')

    def test_flagged_source(self):
        read, write = os.pipe()
        stream = os.fdopen(write, 'w')
        stream.write('1\n2\n')
        stream.close()
        result = self.run_cli(None, 'd1', '-f', '/dev/fd/{}'.format(read))
        self.assertEquals(result, b'1.0\n2.0\n')

    def test_sources(self):
        with self.assertRaises(ValueError):
            self.run_cli(io.BytesIO(b'-1\n'), '-f', '/dev/null', 'd1 + d2', '/dev/null')

    def test_code(self):
        result = self.run_cli(None, 'd', '--code')
        self.assertEquals(result, b'd\n')

    def test_null(self):
        result = self.run_cli(io.BytesIO(b'-1\n'), '-n', 'd')
        self.assertEquals(result, b'')

    def test_named_sources(self):
        one_read, one_write = os.pipe()
        two_read, two_write = os.pipe()

        one_stream = os.fdopen(one_write, 'w')
        two_stream = os.fdopen(two_write, 'w')

        one_stream.write('1\n2\n')
        one_stream.close()
        two_stream.write('10\n20\n')
        two_stream.close()

        one_file = '/dev/fd/{}'.format(one_read)
        two_file = '/dev/fd/{}'.format(two_read)

        result = self.run_cli(
            None,
            '--name', "one", one_file,
            '--name', "two", two_file,
            'one + two')

        self.assertEquals(result, b'11.0\n22.0\n')
