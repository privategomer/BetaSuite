#!/usr/bin/env python3
"""
machine_config.py - the old name of tools/setup/local_config.py.

The machine config (../resources/betaconfig_machine.py) was replaced by
the local config (betaconfig_local.py beside betaconfig.py). This name
forwards every argument to local_config.py, after one line saying so,
so setup scripts and habits written for it keep working.
"""

import os
import sys

sys.path.insert( 0, os.path.dirname( os.path.abspath( __file__ ) ) )

import local_config

NOTICE = ( "machine_config.py is now local_config.py, and it edits betaconfig_local.py; "
           "this name forwards to it and will be removed in a later release" )

if __name__ == '__main__':
    sys.exit( local_config.main( notice=NOTICE ) )
