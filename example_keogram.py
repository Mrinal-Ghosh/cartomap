import numpy as np
# from shapely import geometry as sgeom
# from copy import copy
from datetime import datetime
import apexpy as ap
import matplotlib.pyplot as plt
import cartomap as cm
import h5py
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

if __name__ == '__main__':
    # use local file location
    fn = 'C:\\Users\\mrina\\Desktop\\data\\conv_0428T0000-0429T0000.h5'
    with h5py.File(fn, 'r') as f:
        lat = f['GPSTEC']['lat']
        lon = f['GPSTEC']['lon']
        t = f['GPSTEC']['time']
        im = f['GPSTEC']['im']

        A = ap.Apex(date=t[0])
        fig, axes = plt.subplots(2,2)
        cm.plotKeogram(im, t, latline=51, mlon_levels=[20, 50, 80], meridians=[30, 60, 90], apex=False, geo=False, ax=axes[0][0])
        cm.plotKeogram(im, t, latline=14, mlon_levels=[-20, 50, 80], meridians=[30, 60, 90], apex=False, geo=True, ax=axes[0][1])
        cm.plotKeogram(im, t, lonline=0, mlat_levels=[20, -50, 80], parallels=[-30, -60, -150], apex=True, geo=False, ax=axes[1][0])
        cm.plotKeogram(im, t, lonline=70, mlat_levels=[20, -50, 80], parallels=[-30, -60, 90], apex=True, geo=True, ax=axes[1][1])
        plt.show()
