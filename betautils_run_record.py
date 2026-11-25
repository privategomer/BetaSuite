"""
betautils_run_record.py - what happened to one file in a run, and the
one place that hands it to everything that keeps it.

A RunRecord holds the facts of one file's run: which file and content,
when, the outcome and why, the output and its keys, the video type and
the rule that chose it, the error if it failed. The pipeline fills it as
it goes and gives it to a RunRecorder, which hands it to each of its
sinks:

  StatsLogSink     the stats row (betautils_log.write_stats), as always
  FailureLogSink   the failure row (betautils_log.write_failure)
  catalog history  betautils_catalog_history.HistorySink, when the
                   front end opened the catalog

Each sink owns what it does with a record; none reads another's output
back. The logs stay the logs (for troubleshooting and the analysis
tools); the catalog's history is its own table, written from the same
record.

Two kinds of sink. A REQUIRED sink's error goes to the caller as it
always did: a stats row that cannot be written because the disk is full
still turns the file into a disk.full stop. A best-effort sink's error
is logged and ends there: the catalog never fails a run.
"""

import time


class RunRecord:
    """
    One file's run. Attributes, all optional until known:

      program      'betatv' or 'engine'
      job          the job id
      path         the source as the run read it
      hash         its content hash (None when it could not be read)
      started_at   wall-clock time.time() the file started
      ended_at     when the record was made
      outcome      processed, skipped, failed or out_of_space
      reason       why, as a code: output.present, source.not_video,
                   source.unreadable, file.failed, disk.full; None when
                   processed
      output       the final output's path
      keys         { detection, censor, encode }
      profile, rule, rule_params, sample_fps   the video type and why
      preview      True for a preview slice
      error_type, stage                        a failure's Python type and stage
      stats        the stats row, for a processed file
      failure      the failure row, for a failed one
      moved_to     where move_completed_sources put the source after
    """

    def __init__( self, path, program=None, job=None, clock=time.time ):
        self.program = program
        self.job = job
        self.path = path
        self.hash = None
        self.started_at = clock()
        self.ended_at = None
        self.outcome = None
        self.reason = None
        self.output = None
        self.keys = None
        self.profile = None
        self.rule = None
        self.rule_params = {}
        self.sample_fps = None
        self.preview = False
        self.error_type = None
        self.stage = None
        self.stats = None
        self.failure = None
        self.moved_to = None
        self._clock = clock

    def set_paths( self, paths ):
        """The output path and keys from betatv._build_paths."""
        if not paths:
            return
        self.output = paths.get( 'final' )
        self.keys = { 'detection': paths.get( 'detection_key' ), 'censor': paths.get( 'censor_key' ),
                      'encode': paths.get( 'encode_key' ) }
        self.sample_fps = paths.get( 'sample_fps', self.sample_fps )

    def set_profile( self, profile, reason ):
        self.profile = profile
        self.rule = getattr( reason, 'code', None )
        self.rule_params = dict( getattr( reason, 'params', {} ) or {} )

    def end( self, outcome, reason=None ):
        self.outcome = outcome
        self.reason = reason
        self.ended_at = self._clock()
        return self

    def __repr__( self ):
        return 'RunRecord(%r, %s, %s)'%( self.path, self.outcome, self.reason )


class StatsLogSink:
    """The stats row of a processed file, exactly as betatv wrote it."""

    required = True

    def record( self, run ):
        if run.stats is not None:
            import betautils_log as bu_log
            bu_log.write_stats( run.stats )


class FailureLogSink:
    """The failure row of a file that did not finish."""

    required = True

    def record( self, run ):
        if run.failure is not None:
            import betautils_log as bu_log
            bu_log.write_failure( run.failure )


class RunRecorder:
    """
    Hands each RunRecord to every sink, in order. A sink is any object
    with record( run ), an optional moved( run ), and a `required` flag.
    """

    def __init__( self, sinks, program=None, logger=None ):
        self.sinks = list( sinks )
        self.program = program
        self.logger = logger

    def start( self, path, job=None ):
        return RunRecord( path, program=self.program, job=job )

    def record( self, run ):
        for sink in self.sinks:
            self._call( sink, 'record', run )

    def moved( self, run, new_path ):
        """The source of a recorded run was moved (move_completed_sources)."""
        run.moved_to = new_path
        for sink in self.sinks:
            if hasattr( sink, 'moved' ):
                self._call( sink, 'moved', run )

    def _call( self, sink, method, run ):
        if getattr( sink, 'required', False ):
            getattr( sink, method )( run )
            return
        try:
            getattr( sink, method )( run )
        except Exception:
            if self.logger is not None:
                self.logger.warning( "could not keep the record of %s in %s"%(
                    run.path, type( sink ).__name__ ), exc_info=True )


def log_recorder( program=None, logger=None ):
    """The recorder every run has had: the stats and failure logs."""
    return RunRecorder( [ StatsLogSink(), FailureLogSink() ], program=program, logger=logger )
