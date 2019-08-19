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

        cm.plotSlice(im=im, t=t, time=23, lonline=0, magnetic=True, average=True, conjugate=False)

        """
        Parameters:
        im - im from conv*.h5
        t - t from conv*.h5
        time - slide number ([0,240] usually)
        lonline, latline - value in
        magnetic - true for apex, false/blank for geo
        ax - axis to be plotted on
        average - mean of neighbouring two lat/lon
        skip - # readings to skip
        conjugate - magnetic conjugate mapped to empty squares
        figsize - tuple (w,h)
        height - (currently bugged with inconsistent capped value) - height for apexpy
        """

        plt.show()
