import sys
import os
import lyse
import numpy as np
import h5py
from fake_result import fake_result

# Add analysis lib to path
ROOT_PATH = r"X:\userlib\analysislib"
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)
from analysis.data import h5lyze as hz

df = lyse.data()
h5_path = df.filepath.iloc[-1]

# Your analysis on the lyse DataFrame goes here
if len(df):
    ix = df.iloc[-1].name[0]
    subdf = df.loc[ix]
    your_result = fake_result(subdf.x.mean())
    your_condition = len(subdf) > 3
    with h5py.File(h5_path, mode='r+') as f:
        globals_dict = hz.attributesToDictionary(f['globals'])
        info_dict = hz.getAttributeDict(f)
    your_condition = (info_dict.get('run number') % 2 == 1)

    # Save sequence analysis result in latest run
    run = lyse.Run(h5_path=df.filepath.iloc[-1])
    run.save_result(name='y', value=your_result if your_condition else np.nan)
