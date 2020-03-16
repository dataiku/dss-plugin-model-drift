"""
Do not keep this file in the final plugin.

This file is used in order to have

"""

import joblib
import os
import pdb

OBJECT_PATHS = "/Users/thibaultdesfontaines/devenv/dss-home/plugins/dev/model-drift/interactive-sandbox/objects/"

import sys

path_to_doctor = "/Users/thibaultdesfontaines/devenv/dip/src/main/python/"
sys.path.insert(0, path_to_doctor)

import dataiku

def play_model_accessor():
    model_accessor = joblib.load(OBJECT_PATHS+'model_accessor')
    dir(model_accessor)
    pass

if __name__ == '__main__':
    print(play_model_accessor())