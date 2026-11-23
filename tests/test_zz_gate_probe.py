"""Throwaway: fails on Windows only, to show the CI gate blocks. Never merged."""
import os
import unittest


class TestGateProbe( unittest.TestCase ):

    def test_fails_on_windows( self ):
        self.assertNotEqual( os.name, 'nt' )
