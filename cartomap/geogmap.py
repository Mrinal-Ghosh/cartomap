#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 10:32:04 2018
@author: smrak
"""
import numpy as np
# from shapely import geometry as sgeom
# from copy import copy
from datetime import datetime
import apexpy as ap
import igrf12
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature.nightshade import Nightshade
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.path as mpath
import matplotlib.dates as mdates
# from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from pandas.plotting import register_matplotlib_converters
import warnings

register_matplotlib_converters()

theta = np.linspace(0, 2 * np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)


def plotCartoMap(latlim=None, lonlim=None, parallels=None, meridians=None,
                 pole_center_lon=0, figsize=(12, 8), terrain=False, ax=False,
                 projection='stereo', title='', resolution='110m',
                 states=True, grid_linewidth=0.5, grid_color='black',
                 grid_linestyle='--', background_color=None, border_color='k',
                 figure=False, nightshade=False, ns_dt=None, ns_alpha=0.1,
                 apex=False, igrf=False, date=None,
                 mlat_levels=None, mlon_levels=None, alt_km=0.0,
                 mlon_colors='blue', mlat_colors='red', mgrid_width=1,
                 mgrid_labels=True, mgrid_fontsize=12, mlon_cs='mlon',
                 incl_levels=None, decl_levels=None, igrf_param='incl',
                 mlon_labels=True, mlat_labels=True, mgrid_style='--',
                 label_colors='k',
                 decl_colors='k', incl_colors='k'):

    if lonlim is None:
        if projection in ['southpole', 'northpole']:
            lonlim = [-180, 180]
        else:
            lonlim = [-40, 40]
    if latlim is None:
        if projection is'southpole':
            latlim = [-90, 0]
        elif projection is 'northpole':
            latlim = [90, 0]
        else:
            latlim = [0, 75]

    STATES = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',
        facecolor='none')

    if not ax:
        if figsize is None:
            fig = plt.figure()
        else:
            fig = plt.figure(figsize=figsize)

        if projection == 'stereo':
            ax = plt.axes(projection=ccrs.Stereographic(central_longitude=(sum(lonlim) / 2)))
        elif projection == 'merc':
            ax = plt.axes(projection=ccrs.Mercator())
        elif projection == 'plate':
            ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=(sum(lonlim) / 2)))
        elif projection == 'lambert':
            ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=(sum(lonlim) / 2),
                                                           central_latitude=(sum(latlim) / 2)))
        elif projection == 'mollweide':
            ax = plt.axes(projection=ccrs.Mollweide(central_longitude=(sum(lonlim) / 2)))
        elif projection == 'northpole':
            ax = plt.axes(projection=ccrs.NorthPolarStereo())
            ax.set_boundary(circle, transform=ax.transAxes)
            # polarLim(ax, projection)
        elif projection == 'southpole':
            ax = plt.axes(projection=ccrs.SouthPolarStereo())
            ax.set_boundary(circle, transform=ax.transAxes)
            # polarLim(ax, projection)
    if background_color is not None:
        ax.background_patch.set_facecolor(background_color)
    ax.set_title(title)
    ax.coastlines(color=border_color, resolution=resolution)  # 110m, 50m or 10m
    if states:
        ax.add_feature(STATES, edgecolor=border_color)
    ax.add_feature(cfeature.BORDERS, edgecolor=border_color)
    if terrain:
        ax.stock_img()
    if nightshade:
        assert ns_dt is not None
        assert ns_alpha is not None
        ax.add_feature(Nightshade(ns_dt, ns_alpha))
    # Draw Parallels
    if projection == 'merc' or projection == 'plate':
        if isinstance(meridians, np.ndarray):
            meridians = list(meridians)
        if isinstance(parallels, np.ndarray):
            parallels = list(parallels)

        gl = ax.gridlines(crs=ccrs.PlateCarree(), color=grid_color, draw_labels=False,
                          linestyle=grid_linestyle, linewidth=grid_linewidth)
        if meridians is None:
            gl.xlines = False
        else:
            if len(meridians) > 0:
                gl.xlocator = mticker.FixedLocator(meridians)
                gl.xlabels_bottom = True
            else:
                gl.ylines = False

        if parallels is None:
            gl.ylines = False
        else:
            if len(parallels) > 0:
                gl.ylocator = mticker.FixedLocator(parallels)
                #                ax.yaxis.set_major_formatter(LONGITUDE_FORMATTER)
                gl.ylabels_left = True
            else:
                gl.ylines = False
    else:
        gl = ax.gridlines(crs=ccrs.PlateCarree(), color=grid_color, draw_labels=False,
                          linestyle=grid_linestyle, linewidth=grid_linewidth)
        if meridians is None:
            gl.xlines = False
        else:
            #            if isinstance(meridians, np.ndarray):
            #                meridians = list(meridians)
            #            ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
            #            ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)
            #            if isinstance(meridians, np.ndarray):
            #                meridians = list(meridians)
            gl.xlocator = mticker.FixedLocator(meridians)
        #        else:
        #            gl.xlines = False
        if parallels is not None:
            if isinstance(parallels, np.ndarray):
                parallels = list(parallels)
            gl.ylocator = mticker.FixedLocator(parallels)
        else:
            gl.ylines = False

    # Geomagnetic coordinates @ Apex
    if apex:
        if date is None:
            date = datetime(2017, 12, 31, 0, 0, 0)
        assert isinstance(date, datetime)

        A = ap.Apex(date=date)
        # Define levels and ranges for conversion
        if mlon_cs == 'mlt':
            if mlon_levels is None:
                mlon_levels = np.array([])
                mlon_range = np.arange(0, 24.1, 0.1)
            elif isinstance(mlon_levels, bool):
                if mlon_levels == False:
                    mlon_levels = np.array([])
                    mlon_range = np.arange(0, 24.1, 0.1)
            else:
                mlon_range = np.arange(mlon_levels[0], mlon_levels[-1] + 0.1, 0.1)
        else:
            if mlon_levels is None:
                mlon_levels = np.array([])
                mlon_range = np.arange(-180, 180, 0.5)
            elif isinstance(mlon_levels, bool):
                if mlon_levels == False:
                    mlon_levels = np.array([])
                    mlon_range = np.arange(-180, 181, 0.1)
            else:
                mlon_range = np.arange(mlon_levels[0], mlon_levels[0] + 362, 0.1)
        if mlat_levels is None:
            mlat_levels = np.arange(-90, 90.1, 5)
        mlat_range = np.arange(mlat_levels[0], mlat_levels[-1] + 0.1, 0.1)
        # Do meridans
        for mlon in mlon_levels:
            MLON = mlon * np.ones(mlat_range.size)
            if mlon_cs == 'mlt':
                y, x = A.convert(mlat_range, MLON, 'mlt', 'geo', datetime=date)
            else:
                y, x = A.convert(mlat_range, MLON, 'apex', 'geo')
            # Plot meridian
            inmap = np.logical_and(x >= lonlim[0], x <= lonlim[1])
            if np.sum(inmap) > 10:
                ax.plot(np.unwrap(x, 180), np.unwrap(y, 90), c=mlon_colors,
                        lw=mgrid_width, linestyle=mgrid_style,
                        transform=ccrs.PlateCarree())

            # Labels
            if mlon_labels:
                ix = np.argmin(abs(y - np.mean(latlim)))
                mx = x[ix] - 1 if mlon >= 10 else x[ix] - 0.5
                my = np.mean(latlim)
                if np.logical_and(mx >= lonlim[0], mx <= lonlim[1]):
                    if mlon_cs == 'mlt' and mlon != 0:
                        ax.text(mx, my, str(int(mlon)), color=label_colors,
                                fontsize=14, backgroundcolor='white',
                                transform=ccrs.PlateCarree())
                    elif mlon_cs != 'mlt' and mlon != 360:
                        ax.text(mx, my, str(int(mlon)), color=label_colors,
                                fontsize=14, backgroundcolor='white',
                                transform=ccrs.PlateCarree())
        # Do parallels
        for mlat in mlat_levels:
            MLAT = mlat * np.ones(mlon_range.size)
            if mlon_cs == 'mlt':
                gy, gx = A.convert(MLAT, mlon_range, 'mlt', 'geo', datetime=date)
            else:

                gy, gx = A.convert(MLAT, mlon_range, 'apex', 'geo', datetime=date)
            inmap = np.logical_and(gy >= latlim[0], gy <= latlim[1])
            if np.sum(inmap) > 2:
                ax.plot(np.unwrap(gx, 180), np.unwrap(gy, 90), c=mlat_colors,
                        lw=mgrid_width, linestyle=mgrid_style,
                        transform=ccrs.PlateCarree())

            # Labels
            if mlat_labels:
                ix = np.argmin(abs(gx - np.mean(lonlim)))
                mx = np.mean(lonlim)
                my = gy[ix] - 0.5
                if np.logical_and(mx >= lonlim[0], mx <= lonlim[1]) and \
                        np.logical_and(my >= latlim[0], my <= latlim[1]):
                    ax.text(mx, my, str(int(mlat)), color=label_colors,
                            fontsize=14, backgroundcolor='white',
                            transform=ccrs.PlateCarree())

    if igrf:
        glon = np.arange(lonlim[0] - 40, lonlim[1] + 40.1, 0.5)
        glat = np.arange(-90, 90 + 0.1, 0.5)
        longrid, latgrid = np.meshgrid(glon, glat)
        mag = igrf12.gridigrf12(t=date, glat=latgrid, glon=longrid, alt_km=alt_km)
        if decl_levels is not None:
            z = mag.decl.values
            for declination in decl_levels:
                ax.contour(longrid, latgrid, z, levels=declination,
                           colors=decl_colors, transform=ccrs.PlateCarree())
        if incl_levels is not None:
            z = mag.incl.values
            for inclination in incl_levels:
                ax.contour(longrid, latgrid, z, levels=declination,
                           colors=incl_colors, transform=ccrs.PlateCarree())
    # Set Extent

    ax.set_extent([lonlim[0], lonlim[1], latlim[0], latlim[1]], crs=ccrs.PlateCarree())

    if 'fig' in locals():
        return fig, ax
    else:
        return ax


def plotKeogram(im=None, t=None, latline=None, lonline=None, line=None, skip=None, average=False, magnetic=False, parallels=None,
                meridians=None, mlat_levels=None, mlon_levels=None, ax=False, figsize=None,
                conjugate=True, height=350):

    # pass entire im and t from .h5 file
    # for mlat or mlon set linetype = 'm'

    def conjugate_img(img):
        nanind = np.where(np.isnan(img))
        nanindices = list(zip(nanind[0], nanind[1]))
        conj = img
        for nanindex in nanindices:
            conj[nanindex] = np.flipud(img)[nanindex]
        return conj

    im = np.array(im)
    A = ap.Apex(date=t[0])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if magnetic is False:  # geo
            if latline is None:
                y = range(-90, 90)
                lonline += 180
                if average:
                    image = np.nanmean(np.array([[im[0:, lonline % 360, 0:]], [im[0:, ((lonline+1) % 360), 0:]],
                                       [im[0:, ((lonline-1) % 360), 0:]]]), axis=0)
                    image = np.squeeze(image)  # remove singular dimension
                else:
                    image = np.array(im)[0:, lonline % 360, 0:]
            elif lonline is None:
                y = range(-180, 180)
                latline += 90
                if average:
                    image = np.nanmean(np.array([[im[0:, 0:, latline % 180]], [im[0:, 0:, (latline+1) % 180]],
                                       [im[0:, 0:, (latline-1) % 180]]]), axis=0)
                    image = np.squeeze(image)
                else:
                    image = np.array(im)[0:, 0:, latline % 180]
            image = np.flipud(np.rot90(image))

        else:  # magnetic
            indexmake = np.vectorize(lambda x: int(round(x)))
            image = np.array([])
            mask = np.zeros((im.shape[2], im.shape[1]), dtype=bool)

            if latline is None:  # longitude
                y = range(-90, 90)
                lat_ind, lon_ind = np.round(A.convert(range(-90, 90), lonline, 'apex', 'geo'))
                indices = indexmake(np.array(list(zip(lat_ind + 90, lon_ind + 180))))
                indices = list(map(tuple, indices))

                for index in indices:
                    mask[index[0] % 180, index[1] % 360] = True
                mask = np.rot90(mask)

                if average:
                    for index in indices:
                        image = np.append(image, np.nanmean([[im[0:, index[1] % 360, index[0] % 180]],
                                                            [im[0:, (index[1] + 1) % 360, index[0] % 180]],
                                                            [im[0:, (index[1] - 1) % 360, index[0] % 180]]], axis=0))
                else:
                    for index in indices:
                        image = np.append(image, im[0:, index[1], index[0]])

                image = np.reshape(image, (len(image) // len(t), len(t)))

                if conjugate:
                    image = conjugate_img(image)

            elif lonline is None:  # latitude
                lat_ind, lon_ind = np.round(A.convert(latline, range(-180, 180), 'apex', 'geo'))
                indices = indexmake(np.array(list(zip(lat_ind + 90, lon_ind + 180))))
                indices = list(map(tuple, indices))
                y = range(-180, 180)

                for index in indices:
                    mask[index[0] % 180, index[1] % 360] = True

                mask = np.rot90(mask)

                if average:
                    for index in indices:
                        image = np.append(image, np.nanmean([[im[0:, index[1] % 360, index[0] % 180]],
                                                             [im[0:, index[1] % 360, (index[0] + 1) % 180]],
                                                             [im[0:, index[1] % 360, (index[0] - 1) % 180]]], axis=0))
                else:
                    for index in indices:
                        image = np.append(image, im[0:, index[1] % 360, index[0] % len(t)])
                image = np.reshape(image, (len(image) // len(t), len(t)))

    t = list(map(datetime.fromtimestamp, t))
    mt = mdates.date2num((t[0], t[-1]))

    if skip is not None:
        image = image[::skip]

    if not ax:
        if figsize is None:
            fig = plt.figure()
        else:
            fig = plt.figure(figsize=figsize)
        ax = fig.gca()

    # fig.suptitle(f'{t[0].strftime("%Y-%m-%d")}')
    ax.imshow(image, extent=[mt[0], mt[1], y[0], y[-1]], aspect='auto', cmap='gist_ncar')

    if latline is None:
        if mlat_levels is not None:
            for level in mlat_levels:
                glat, _ = A.convert(level, lonline, 'geo', 'apex', height=height)
                ax.axhline(glat, linestyle='--', linewidth=1, color='firebrick')
        if parallels is not None:
            for level in parallels:
                ax.axhline(level, linestyle='--', linewidth=1, color='cornflowerblue')
    elif lonline is None:
        if mlon_levels is not None:
            for level in mlon_levels:
                _, glon = A.convert(latline, level, 'geo', 'apex', height=height)
                ax.axhline(glon, linestyle='--', linewidth=1, color='firebrick')
        if meridians is not None:
            for level in meridians:
                ax.axhline(level, linestyle='--', linewidth=1, color='cornflowerblue')

    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    # fig.autofmt_xdate()

    if 'fig' in locals():
        return fig, ax
    else:
        return ax
