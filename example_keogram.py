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

        cm.plotKeogram(im=im, t=t, lonline=0, magnetic=True, mlat_levels=[20, 50, 80], parallels=[30, 60, 90],
                       ax=ax1, average=True, conjugate=True)
        """
        Parameters:
        im - im from conv*.h5
        t - t from conv*.h5
        lonline, latline - value in
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
                        latlim=[-90, 90], lonlim=[-180, 180], ax=ax2, meridians=[-51], parallels=[30, 60, 90],
                        mlat_levels=[20, 50, 80], mlat_labels=False)

        msh = ax2.pcolormesh(lon, lat, np.transpose(im[0:][0:][0]), transform=ccrs.PlateCarree(), cmap='gist_ncar')
        fig.colorbar(msh, label='Total Electron Concentration [TECu]')

        # cm.plotKeogram(im, t, latline=14, mlon_levels=[-20, 50, 80], meridians=[30, 60, 90], apex=False, geo=True, ax=axes[0][1])
        # cm.plotKeogram(im, t, lonline=0, mlat_levels=[20, -50, 80], parallels=[-30, -60, -150], apex=True, geo=False, ax=axes[1][0])
        # cm.plotKeogram(im, t, lonline=70, mlat_levels=[20, -50, 80], parallels=[-30, -60, 90], apex=True, geo=True, ax=axes[1][1])

        plt.show()
