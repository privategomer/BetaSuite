import numpy as np
from pathlib import Path
import os
import subprocess
import math
import sys
import hashlib
import json
from tkinter import *
import timeit
from multiprocessing import shared_memory
import cv2
import mss
import onnxruntime

import sys as _sys, os as _os
_sys.path.insert( 0, _os.path.dirname( _os.path.dirname( _os.path.dirname( _os.path.abspath( __file__ ) ) ) ) )  # so 'import betaconfig' etc still resolve from tools/analysis/

import betaconfig

sct = mss.mss()
monitor = {'top':0,'left':0,'width':320,'height':320,'mon':1}
image = np.array(sct.grab(monitor))[:,:,:3]

image = image.astype( np.float32 )
image -= [ 103.939, 116.779, 123.68 ]

if betaconfig.gpu_enabled:
    providers = [ ( 'CUDAExecutionProvider', { 'device_id':0 } ) ]
else:
    providers = [ ( 'CPUExecutionProvider', {} ) ]

options = onnxruntime.SessionOptions()
options.log_severity_level = 0

session = onnxruntime.InferenceSession( '../../../resources/model/detector_v2_default_checkpoint.onnx', providers=providers )

model_outputs = [ s_i.name for s_i in session.get_outputs() ]
model_input_name = session.get_inputs()[0].name

outputs = session.run( model_outputs, {model_input_name: np.expand_dims( image, axis=0) } )
