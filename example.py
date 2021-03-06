#!/usr/bin/env python3
import cartomap as cm
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib import path as mpath
import numpy as np


fig = plt.figure()
ax1 = fig.add_subplot(121, projection=ccrs.NorthPolarStereo())
cm.plotCartoMap(projection='northpole', terrain=True, apex=True, igrf=True, mlon_cs='mlt', latlim=[0,90], lonlim=[-180,180], ax=ax1, mlat_levels=[0,20,40,60,80], mlat_labels=False)


ax2 = fig.add_subplot(122, projection=ccrs.SouthPolarStereo())
cm.plotCartoMap(projection='southpole', terrain=True, apex=True, igrf=True, mlon_cs='mlt', latlim=[-90, -30], lonlim=[-180, 180], ax=ax2, mlat_levels=[0,-20,-40,-60,-80], mlat_labels=False)

# ny_lon, ny_lat = -74.00, 40.71
# delhi_lon, delhi_lat = 77.23, 28.61
#
# plt.plot([ny_lon, delhi_lon], [ny_lat, delhi_lat],
#          color='blue', linewidth=2, marker='x',
#          transform=ccrs.Geodetic()
#          )

plt.show()

######################### Supported Arguments ##############################
# latlim, # Latitude limits
# lonlim, # Longitude limits
# parallels, # Specify parallels to draw
# meridians, # Specify meridians to draw
# figsize, # Define figure size -> plt.figure(figsize=figsize)
# projection, # Projection type, Look below
# title, # Figure title
# resolution, # As per CartoPy, three options are possible 110m, 50m or 10m
# states, # Draw states
# grid_linewidth, # Grid == meridians&parallels
# grid_color,
# grid_linestyle,
# terrain, # Orographic colormap, defaults as it comes with CartoPy
# background_color,
# border_color='k'. # Border=states and countries

# projections
# Sterographic as 'stereo',
# Mercator as 'merc',
# PlateCarree as 'plate',
# LambertConformal as 'lambert'
# Mollweide as 'mollweide'
# NorthPolarStereo as 'northpole'
# SouthPolarStereo as 'southpole'
