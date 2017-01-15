
# make code as python 3 compatible as possible
from __future__ import absolute_import, division, print_function, unicode_literals

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
        self.run_cli(io.BytesIO(b'1\n2\n3\n'), '--repr',  '(d,)')
        self.run_cli(io.BytesIO(b'1\n2\n3\n'), '--repr',  '[d]')
        self.run_cli(io.BytesIO(b'1\n2\n3\n'), '--repr',  'd + 1')
        self.run_cli(io.BytesIO(b'1\n2\n3\n'), '--repr',  'd ** 2')
        self.run_cli(io.BytesIO(b'1\n2\n3\n'), '--repr',  'd.sum()')
