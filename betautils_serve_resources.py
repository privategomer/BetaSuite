"""
betautils_serve_resources.py - the resource commands of serve mode.

  hw.detect   this machine (betautils_hardware.machine) and the settings
              its facts settle without measuring (safe_settings)

The governor, performance evidence, Optimize and storage commands join
this table as they land.
"""


def hw_detect( _server, _data ):
    import betaconfig
    import betautils_hardware as bu_hardware
    facts = bu_hardware.machine()
    facts['safe_settings'] = bu_hardware.safe_settings( facts, betaconfig )
    return facts


HANDLERS = {
    'hw.detect': hw_detect,
}
