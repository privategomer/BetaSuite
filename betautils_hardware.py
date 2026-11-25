"""
betautils_hardware.py - what this machine can run, read without loading a model.

Small and stdlib-only at import time (onnxruntime is imported inside the
call that needs it), so the engine and the setup tools can ask cheaply.
Hardware detection for the resource governor grows here.

A usable NVIDIA GPU needs both halves: onnxruntime-gpu lists
CUDAExecutionProvider even on a machine with no GPU, and nvidia-smi can
see a GPU that a CPU-only onnxruntime cannot use.
"""

import subprocess

CUDA_PROVIDER = 'CUDAExecutionProvider'


def _run_quiet( command, timeout=30 ):
    """( exit code, stdout + stderr ), or ( None, why ) when it could not run."""
    try:
        done = subprocess.run( command, capture_output=True, text=True, timeout=timeout )
    except ( OSError, subprocess.SubprocessError ) as err:
        return None, str( err )
    return done.returncode, ( done.stdout or '' ) + ( done.stderr or '' )


def nvidia_smi( fields, run=_run_quiet ):
    """{ field: value } for the first GPU, or {} without nvidia-smi."""
    code, text = run( [ 'nvidia-smi', '--query-gpu=%s'%( ','.join( fields ), ),
                        '--format=csv,noheader,nounits' ] )
    if code != 0 or not text:
        return {}
    values = [ part.strip() for part in text.strip().splitlines()[0].split( ',' ) ]
    return dict( zip( fields, values ) )


def onnx_providers():
    """onnxruntime's available execution providers, or () when it is not installed."""
    try:
        import onnxruntime
    except ImportError:
        return ()
    try:
        return tuple( onnxruntime.get_available_providers() )
    except Exception:
        return ()


def cuda_usable( providers, gpu ):
    """True when onnxruntime can use CUDA and nvidia-smi sees a GPU."""
    return CUDA_PROVIDER in ( providers or () ) and bool( gpu )


def detect():
    """
    This machine's GPU facts, as plain data:
    { 'providers': [...], 'gpu': { 'name': ... } or None, 'cuda_usable': bool }.
    """
    providers = onnx_providers()
    gpu = nvidia_smi( [ 'name' ] ) or None
    return { 'providers': list( providers ), 'gpu': gpu, 'cuda_usable': cuda_usable( providers, gpu ) }
