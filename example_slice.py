import numpy as np
import apexpy as ap
import matplotlib.pyplot as plt
import cartomap as cm
import h5py
from pandas.plotting import register_matplotlib_converters
import cartopy.crs as ccrs

register_matplotlib_converters()

if __name__ == '__main__':
    # use local file location
    fn = 'C:\\Users\\mrina\\Desktop\\data\\conv_0428T0000-0429T0000.h5'
    with h5py.File(fn,'r') as f:
        lat = f['GPSTEC']['lat']
        lon = f['GPSTEC']['lon']
        t = f['GPSTEC']['time']
        im = f['GPSTEC']['im']

        fig = plt.figure()
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212)
        cm.plotSlice()