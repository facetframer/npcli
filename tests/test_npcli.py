# make code as python 3 compatible as possible
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import io
import tempfile
import unittest

from npcli.npcli import run

class NpcliFeaturesTest(unittest.TestCase):
    def setUp(self):
        self.direc = tempfile.mkdtemp()

    def run_cli(self, stdin, *args):
        result = run(stdin, args)
        return b''.join(result)

    def test_kitchen(self):
        result = self.run_cli(io.BytesIO(b'-1\n'), '-K', 'abs(d)')
        self.assertEquals(result, b'1.0\n')
