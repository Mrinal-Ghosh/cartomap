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
    with h5py.File(fn, 'r') as f:
        lat = f['GPSTEC']['lat']
        lon = f['GPSTEC']['lon']
        t = f['GPSTEC']['time']
        im = f['GPSTEC']['im']

        A = ap.Apex(date=t[0])
        fig = plt.Figure()
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212, projection=ccrs.PlateCarree())

        cm.plotKeogram(im=im, t=t, latline=0, magnetic=False, mlat_levels=[-90, -60, -30, 0, 30, 60, 90], parallels=[],
                       ax=ax1, average=True, conjugate=False)
        """
        Parameters:
        im - im from conv*.h5
        t - t from conv*.h5
        lonline, latline - value in apex or geo coordinates
        line - (currently not working) arbitrary line of form [lat_1, lon_1, lat_2, lon_2]
        magnetic - true for apex, false/blank for geo
        mlat_levels, mlon_levels, parallels, meridians - list
        ax - axis to be plotted on
        average - mean of neighbouring two lat/lon
        skip - # readings to skip
        conjugate - # magnetic conjugate mapped to empty squares
        figsize - tuple (w,h)
        height - (currently bugged with inconsistent capped value) - height for apexpy
        """

        cm.plotCartoMap(projection='plate', terrain=True, apex=True, igrf=True, mlon_cs='mlt',
                        latlim=[-90, 90], lonlim=[-180, 180], ax=ax2, meridians=[-90,0], parallels=[30, 60, 90],
                        mlat_levels=[-90, -60, -30, 0, 30, 60, 90], mlat_labels=True)

        msh = ax2.pcolormesh(lon, lat, np.transpose(im[0:][0:][0]), transform=ccrs.PlateCarree(), cmap='gist_ncar')
        fig.colorbar(msh, label='Total Electron Concentration [TECu]')

        plt.show()
