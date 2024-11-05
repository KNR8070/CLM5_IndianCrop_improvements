#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 00:56:40 2024

@author: knreddy
"""
#%% Loading required Modules
import xarray as xr
import matplotlib.pyplot as plt
import rioxarray 
import geopandas as gpd
import pandas as pd
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib as mpl
import cmcrameri.cm as cmc
import math
import scipy.stats as stats
from statsmodels.formula.api import ols
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
warnings.filterwarnings("ignore")
#%% Descrete colormap ceation
def discrete_cmap(N, base=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    # base = mpl.colormaps[base_cmap]
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)
#%% Defining directories
wrk_dir = '/Users/knreddy/Documents/GMD_Paper/Figures/'
shp_dir = '/Users/knreddy/Documents/CLM_DataAnalysis/CLM5_RunAnalysis_Default_Case/Spyder_Matplotlib/India_Shapefile_'

data_dir = '/Users/knreddy/Documents/CLM_DataAnalysis/CLM5_RunAnalysis_Default_Case/CLM_New_Analysis_Nov23/DataforPlotting/'
    
month = {'1':'Jan','2':'Feb','3':'Mar','4':'Apr','5':'May','6':'Jun',
         '7':'Jul','8':'Aug','9':'Sep','10':'Oct','11':'Nov','12':'Dec'}
#%% Reading data for plotting
data_LH = xr.open_dataset(data_dir+'CLM5_FluxCom_LHData_06-Apr-2024.nc')

data_SH = xr.open_dataset(data_dir+'CLM5_FluxCom_SHData_06-Apr-2024.nc')

India = gpd.read_file(shp_dir+'/india_administrative_outline_boundary.shp', crs="epsg:4326")

lat = data_LH['lat']
lon = data_LH['lon']

LH_Def = data_LH['LH_Def']
LH_Mod1 = data_LH['LH_Mod1']
LH_Mod2 = data_LH['LH_Mod2']
LH_FLUXCOM = data_LH['LH_FLUXCOM']

SH_Def = data_SH['SH_Def']
SH_Mod1 = data_SH['SH_Mod1']
SH_Mod2 = data_SH['SH_Mod2']
SH_FLUXCOM = data_SH['SH_FLUXCOM']
#%% Clipping Data for Indian region
LH_Def.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
LH_Def.rio.write_crs("epsg:4326", inplace=True)
clip_LH_Def = LH_Def.rio.clip(India.geometry, India.crs, drop=True)

LH_Mod1.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
LH_Mod1.rio.write_crs("epsg:4326", inplace=True)
clip_LH_Mod1 = LH_Mod1.rio.clip(India.geometry, India.crs, drop=True)

LH_Mod2.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
LH_Mod2.rio.write_crs("epsg:4326", inplace=True)
clip_LH_Mod2 = LH_Mod2.rio.clip(India.geometry, India.crs, drop=True)

LH_FLUXCOM.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
LH_FLUXCOM.rio.write_crs("epsg:4326", inplace=True)
clip_LH_FLUXCOM = LH_FLUXCOM.rio.clip(India.geometry, India.crs, drop=True)

SH_Def.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
SH_Def.rio.write_crs("epsg:4326", inplace=True)
clip_SH_Def = SH_Def.rio.clip(India.geometry, India.crs, drop=True)

SH_Mod1.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
SH_Mod1.rio.write_crs("epsg:4326", inplace=True)
clip_SH_Mod1 = SH_Mod1.rio.clip(India.geometry, India.crs, drop=True)

SH_Mod2.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
SH_Mod2.rio.write_crs("epsg:4326", inplace=True)
clip_SH_Mod2 = SH_Mod2.rio.clip(India.geometry, India.crs, drop=True)

SH_FLUXCOM.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
SH_FLUXCOM.rio.write_crs("epsg:4326", inplace=True)
clip_SH_FLUXCOM = SH_FLUXCOM.rio.clip(India.geometry, India.crs, drop=True)
#%% Plotting Monthly LH: CLM5 vs FLUXCOM
################## Spatial Plotting ###########################################
###############################################################################
titles = ['FLUXCOM', 'CLM5_Def', 'CLM5_Mod1', 'CLM5_Mod2']

fig3, axes3 = plt.subplots(nrows=6, ncols=4, sharex=True, sharey=True, dpi=600, layout='constrained')
fig3.set_size_inches(8,12)
# fig3.subplots_adjust(top=0.9)
# fig3.suptitle('Monthly LH over India (2000 to 2014)', y=1.0)
# fig3.tight_layout(pad=0.4)

vmn = 0
vmx = 150
N1 = 7

cmap = cmc.batlowW_r  # define the colormap
# extract all colors from the .jet map
# cmaplist = [cmap(i) for i in range(cmap.N)]
cmaplist_2 = [cmap(i) for i in np.arange(0,cmap.N,int(256/N1))]
# create the new map
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist_2[1:], N1)
# cmap[0] = [1,1,1,1.0]
# define the bins and normalize
bounds = np.linspace(vmn, vmx, N1)
norm = mpl.colors.BoundaryNorm(bounds, N1+1)
# month_dim = [10, 11, 0, 1, 2]
# month_names = ['Nov','Dec','Jan','Feb','Mar']
for i_row in range(6):
    for i_col in range(4):
        p_row = i_row
        ax = axes3[p_row,i_col]
        ax.tick_params('both', labelsize=15)
        India.plot(facecolor='gray',edgecolor='black',ax=ax)
        if i_row<6 and i_row!=5:
            if i_col==0:
                plot_data = clip_SH_FLUXCOM[i_row,:,:]
                im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
                ax.set(title='',xlabel='')
                ax.set_ylabel(month[str(i_row+1)]+' \n\n Latitude', fontsize=15)
            elif i_col ==1:
                plot_data = clip_LH_Def[i_row,:,:]
                im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
                ax.set(title='', ylabel='',xlabel='')
            elif i_col ==2:
                plot_data = clip_LH_Mod1[i_row,:,:]
                im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
                ax.set(title='', ylabel='',xlabel='')
            else:
                plot_data = clip_LH_Mod2[i_row,:,:]
                im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
                ax.set(title='', ylabel='',xlabel='')
        elif i_row<6 and i_row==5:
            if i_col==0:
                plot_data = clip_LH_FLUXCOM[i_row,:,:]
                im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
                ax.set(title='')
                ax.set_xlabel('Longitude', fontsize=15)  
                ax.set_ylabel(month[str(i_row+1)]+' \n\n Latitude', fontsize=15)
            elif i_col ==1:
                plot_data = clip_LH_Def[i_row,:,:]
                im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
                ax.set(title='', ylabel='')
                ax.set_xlabel('Longitude', fontsize=15)              
            elif i_col ==2:
                plot_data = clip_LH_Mod1[i_row,:,:]
                im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
                ax.set(title='', ylabel='')
                ax.set_xlabel('Longitude', fontsize=15)
            else:
                plot_data = clip_LH_Mod2[i_row,:,:]
                im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
                ax.set(title='',ylabel='')
                ax.set_xlabel('Longitude', fontsize = 15)

        elif i_row>=6 and i_row==11:
            if i_col==0:
                plot_data = clip_LH_FLUXCOM[i_row,:,:]
                im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
                ax.set(title='', ylabel='')
                ax.set_xlabel('Longitude', fontsize=15)
                ax.yaxis.set_tick_params(labelright=False)
                
            elif i_col ==1:
                plot_data = clip_LH_Def[i_row,:,:]
                im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
                ax.set(title='', ylabel='')
                ax.set_xlabel('Longitude', fontsize=15)
                
            elif i_col ==2:
                plot_data = clip_LH_Mod1[i_row,:,:]
                im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
                ax.set(title='', ylabel='')
                ax.set_xlabel('Longitude', fontsize=15)
            else:
                plot_data = clip_LH_Mod2[i_row,:,:]
                im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
                ax.set(title='')
                ax.set_xlabel('Longitude', fontsize = 15)
                ax.set_ylabel('Latitude\n \n'+month[str(i_row+1)], fontsize=15)
                ax.yaxis.set_tick_params(labelright=True)
                
        elif i_row>=6 and i_row!=11:
            if i_col==0:
                plot_data = clip_LH_FLUXCOM[i_row,:,:]
                im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
                ax.set(title='', ylabel='',xlabel='')
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
                ax.yaxis.set_tick_params(labelright=False)
            elif i_col ==1:
                plot_data = clip_LH_Def[i_row,:,:]
                im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
                ax.set(title='', ylabel='',xlabel='')
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
            elif i_col ==2:
                plot_data = clip_LH_Mod1[i_row,:,:]
                im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
                ax.set(title='', ylabel='',xlabel='')
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
            else:
                plot_data = clip_LH_Mod2[i_row,:,:]
                im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
                ax.set(title='',xlabel='')
                ax.set_ylabel('Latitude\n \n'+month[str(i_row+1)], fontsize=15)
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
                ax.yaxis.set_tick_params(labelright=True)
# if i_row<6:
# ax, _ = mpl.colorbar.make_axes(axes3.ravel().tolist())

# cbar = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, 
#                         spacing='proportional',ticks=bounds, boundaries=bounds, format='%1.1f')
# cbar.set_ticks(np.linspace(0,0.3,4))
# cbar.set_label('LH (kgC/m\u00b2/mon)')
cbar = fig3.colorbar(im,ax=axes3[:,3],shrink=0.5)
cbar.set_label('LH (W/m\u00b2)')
cbar.set_ticks(np.linspace(vmn,vmx,5))

# else:
#             # define the bins and normalize
#             bounds = np.linspace(0, 0.3, 13)
#             norm = mpl.colors.BoundaryNorm(bounds, cmap.N)


#             ax, _ = mpl.colorbar.make_axes(axes3.ravel().tolist())

#             cbar = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, location='left',
#                                     spacing='proportional',ticks=bounds, boundaries=bounds, format='%1.1f')
#             cbar.set_ticks(np.linspace(0,0.3,4))
#             cbar.set_label('(kgC/m\u00b2/mon)')

# if i_row>6:
cbar.remove()
# fig3.savefig(wrk_dir+'Monthly LH FLUXCOM_CLM5 over India (2000 to 2014)_Jul to Dec_3.png', 
#               dpi=600, bbox_inches="tight")

#%% Plotting Monthly Difference LH: CLM5 vs FLUXCOM
################## Spatial Plotting ###########################################
###############################################################################
titles = ['FLUXCOM', 'Def-FLUXCOM', 'Mod1-FLUXCOM', 'Mod2-FLUXCOM']

fig3, axes3 = plt.subplots(nrows=6, ncols=4, sharex=True, sharey=True, dpi=600, layout='constrained')
fig3.set_size_inches(8,12)

vmn1 = 0
vmx1 = 150
N1 = 11

vmn2 = -75
vmx2 = 75
N2 = 16

#### cmap for FLUXCOM data
cmap1 = cmc.batlowW_r  
cmaplist_1 = [cmap1(i) for i in np.arange(0,cmap1.N,int(256/N1))]
cmap1 = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist_1[1:], N1)
bounds1 = np.linspace(vmn1, vmx1, N1)
norm1 = mpl.colors.BoundaryNorm(bounds1, N1+1)
#### cmap for CLM5-FLUXCOM data
cmap2 = cmc.vik  
cmap2 = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', 
                                                    colors =[cmc.vik(40), 
                                                          (1, 1., 1), 
                                                          cmc.vik(220)], 
                                                    N=N2+1)
bounds2 = np.linspace(vmn2, vmx2, N2)
norm2 = mpl.colors.BoundaryNorm(bounds2, N2+1)

# for i_row in range(6,12): #for plotting Jul to Dec maps
for i_row in range(6):
    for i_col in range(4):
        p_row = i_row-6
        ax = axes3[p_row,i_col]
        ax.tick_params('both', labelsize=15)
        India.plot(facecolor='gray',edgecolor='black',ax=ax)
        if i_row<6 and i_row!=5:
            if i_col==0:
                plot_data = clip_LH_FLUXCOM[i_row,:,:]
                im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,
                                 levels=bounds1,cmap=cmap1,vmin=vmn1,vmax =vmx1,extend='both')
                ax.set(title='',xlabel='')
                ax.set_ylabel(month[str(i_row+1)]+' \n\n Latitude', fontsize=15)
            elif i_col ==1:
                plot_data = clip_LH_Def[i_row,:,:]-clip_LH_FLUXCOM[i_row,:,:]
                im2 = ax.contourf(plot_data.lon,plot_data.lat,plot_data,
                                 levels=bounds2,cmap=cmap2,vmin=vmn2,vmax =vmx2,extend='both')
                ax.set(title='', ylabel='',xlabel='')
            elif i_col ==2:
                plot_data = clip_LH_Mod1[i_row,:,:]-clip_LH_FLUXCOM[i_row,:,:]
                im2 = ax.contourf(plot_data.lon,plot_data.lat,plot_data,
                                 levels=bounds2,cmap=cmap2,vmin=vmn2,vmax =vmx2,extend='both')
                ax.set(title='', ylabel='',xlabel='')
            else:
                plot_data = clip_LH_Mod2[i_row,:,:]-clip_LH_FLUXCOM[i_row,:,:]
                im2 = ax.contourf(plot_data.lon,plot_data.lat,plot_data,
                                 levels=bounds2,cmap=cmap2,vmin=vmn2,vmax =vmx2,extend='both')
                ax.set(title='', ylabel='',xlabel='')
        elif i_row<6 and i_row==5:
            if i_col==0:
                plot_data = clip_LH_FLUXCOM[i_row,:,:]
                im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,
                                 levels=bounds1,cmap=cmap1,vmin=vmn1,vmax =vmx1,extend='both')
                ax.set(title='')
                ax.set_xlabel('Longitude', fontsize=15)  
                ax.set_ylabel(month[str(i_row+1)]+' \n\n Latitude', fontsize=15)
            elif i_col ==1:
                plot_data = clip_LH_Def[i_row,:,:]-clip_LH_FLUXCOM[i_row,:,:]
                im2 = ax.contourf(plot_data.lon,plot_data.lat,plot_data,
                                 levels=bounds2,cmap=cmap2,vmin=vmn2,vmax =vmx2,extend='both')
                ax.set(title='', ylabel='')
                ax.set_xlabel('Longitude', fontsize=15)              
            elif i_col ==2:
                plot_data = clip_LH_Mod1[i_row,:,:]-clip_LH_FLUXCOM[i_row,:,:]
                im2 = ax.contourf(plot_data.lon,plot_data.lat,plot_data,
                                 levels=bounds2,cmap=cmap2,vmin=vmn2,vmax =vmx2,extend='both')
                ax.set(title='', ylabel='')
                ax.set_xlabel('Longitude', fontsize=15)
            else:
                plot_data = clip_LH_Mod2[i_row,:,:]-clip_LH_FLUXCOM[i_row,:,:]
                im2 = ax.contourf(plot_data.lon,plot_data.lat,plot_data,
                                 levels=bounds2,cmap=cmap2,vmin=vmn2,vmax =vmx2,extend='both')
                ax.set(title='',ylabel='')
                ax.set_xlabel('Longitude', fontsize = 15)

        elif i_row>=6 and i_row==11:
            if i_col==0:
                plot_data = clip_LH_FLUXCOM[i_row,:,:]
                im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,
                                 levels=bounds1,cmap=cmap1,vmin=vmn1,vmax =vmx1,extend='both')
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
                ax.set(title='', ylabel='')
                ax.set_xlabel('Longitude', fontsize=15)
                ax.yaxis.set_tick_params(labelright=False)
                
            elif i_col ==1:
                plot_data = clip_LH_Def[i_row,:,:]-clip_LH_FLUXCOM[i_row,:,:]
                im2 = ax.contourf(plot_data.lon,plot_data.lat,plot_data,
                                 levels=bounds2,cmap=cmap2,vmin=vmn2,vmax =vmx2,extend='both')
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
                ax.set(title='', ylabel='')
                ax.set_xlabel('Longitude', fontsize=15)
                
            elif i_col ==2:
                plot_data = clip_LH_Mod1[i_row,:,:]-clip_LH_FLUXCOM[i_row,:,:]
                im2 = ax.contourf(plot_data.lon,plot_data.lat,plot_data,
                                 levels=bounds2,cmap=cmap2,vmin=vmn2,vmax =vmx2,extend='both')
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
                ax.set(title='', ylabel='')
                ax.set_xlabel('Longitude', fontsize=15)
            else:
                plot_data = clip_LH_Mod2[i_row,:,:]-clip_LH_FLUXCOM[i_row,:,:]
                im2 = ax.contourf(plot_data.lon,plot_data.lat,plot_data,
                                 levels=bounds2,cmap=cmap2,vmin=vmn2,vmax =vmx2,extend='both')
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
                ax.set(title='')
                ax.set_xlabel('Longitude', fontsize = 15)
                ax.set_ylabel('Latitude\n \n'+month[str(i_row+1)], fontsize=15)
                ax.yaxis.set_tick_params(labelright=True)
                
        elif i_row>=6 and i_row!=11:
            if i_col==0:
                plot_data = clip_LH_FLUXCOM[i_row,:,:]
                im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,
                                 levels=bounds1,cmap=cmap1,vmin=vmn1,vmax =vmx1,extend='both')
                ax.set(title='', ylabel='',xlabel='')
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
                ax.yaxis.set_tick_params(labelright=False)
            elif i_col ==1:
                plot_data = clip_LH_Def[i_row,:,:]-clip_LH_FLUXCOM[i_row,:,:]
                im2 = ax.contourf(plot_data.lon,plot_data.lat,plot_data,
                                 levels=bounds2,cmap=cmap2,vmin=vmn2,vmax =vmx2,extend='both')
                ax.set(title='', ylabel='',xlabel='')
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
            elif i_col ==2:
                plot_data = clip_LH_Mod1[i_row,:,:]-clip_LH_FLUXCOM[i_row,:,:]
                im2 = ax.contourf(plot_data.lon,plot_data.lat,plot_data,
                                 levels=bounds2,cmap=cmap2,vmin=vmn2,vmax =vmx2,extend='both')
                ax.set(title='', ylabel='',xlabel='')
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
            else:
                plot_data = clip_LH_Mod2[i_row,:,:]-clip_LH_FLUXCOM[i_row,:,:]
                im2 = ax.contourf(plot_data.lon,plot_data.lat,plot_data,
                                 levels=bounds2,cmap=cmap2,vmin=vmn2,vmax =vmx2,extend='both')
                ax.set(title='',xlabel='')
                ax.set_ylabel('Latitude\n \n'+month[str(i_row+1)], fontsize=15)
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
                ax.yaxis.set_tick_params(labelright=True)

cbar1 = fig3.colorbar(im,ax=axes3[0:2,-1],cmap=cmap1,shrink=0.5)
cbar1.set_label('LH (kgC/m\u00b2/mon)')
cbar1.set_ticks(np.linspace(vmn1,vmx1,4))

cbar2 = fig3.colorbar(im,ax=axes3[2:,-1],shrink=0.5)
cbar2.set_label('LH (kgC/m\u00b2/mon)')
cbar2.set_ticks(np.linspace(vmn2,vmx2,int((N2+1)/2)))

cbar1.remove()
cbar2.remove()
#%% Plotting colorbars
figcbar,axcbar = plt.subplots(nrows=3,ncols=4)
cbar = figcbar.colorbar(im,ax=axcbar[:,3],shrink=0.8)
cbar.set_label('LH (kgC/m\u00b2/mon)')
cbar.set_ticks(np.linspace(vmn1,vmx1,4))

figcbar2,axcbar2 = plt.subplots(nrows=3,ncols=4)
cbar2 = figcbar2.colorbar(im2,ax=axcbar2[:,3],shrink=0.8)
cbar2.set_label('LH (kgC/m\u00b2/mon)')
cbar2.set_ticks(np.linspace(vmn2,vmx2,7))
#%% PLOTTING ANNUAL MEAN
# fig3.savefig(wrk_dir+'Monthly LH FLUXCOM_CLM5 over India (2000 to 2014)_Jul to Dec_3.png', 
#               dpi=600, bbox_inches="tight")
titles = ['FLUXCOM', 'Def-FLUXCOM', 'Mod1-FLUXCOM', 'Mod2-FLUXCOM']
FLUXCOM_data_annual = np.mean(LH_FLUXCOM,axis=0)
Def_data_annaul = np.mean(clip_LH_Def,axis=0)
Mod1_data_annaul = np.mean(clip_LH_Mod1,axis=0)
Mod2_data_annaul = np.mean(clip_LH_Mod2,axis=0)


fig3, axes3 = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True, dpi=600, layout='constrained')
fig3.set_size_inches(8,2)

vmn1 = 0
vmx1 = 0.3
N1 = 13

vmn2 = -0.2
vmx2 = 0.2
N2 = 8 

#### cmap for FLUXCOM data
cmap1 = cmc.batlowW_r  
cmaplist_1 = [cmap1(i) for i in np.arange(0,cmap1.N,int(256/N1))]
cmap1 = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist_1[1:], N1)
bounds1 = np.linspace(vmn1, vmx1, N1)
norm1 = mpl.colors.BoundaryNorm(bounds1, N1+1)
#### cmap for CLM5-FLUXCOM data
cmap2 = cmc.vik  
cmaplist_2 = [cmap2(i) for i in np.arange(0,cmap2.N,int(256/N2))]
cmap2 = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist_2, N2)
bounds2 = np.linspace(vmn2, vmx2, N2)
norm2 = mpl.colors.BoundaryNorm(bounds2, N2+1)
# month_dim = [10, 11, 0, 1, 2]
# month_names = ['Nov','Dec','Jan','Feb','Mar']
for i_col in range(4):
    ax = axes3[i_col]
    ax.tick_params('both', labelsize=15)
    India.plot(facecolor='gray',edgecolor='black',ax=ax)
    if i_col==0:
        plot_data = FLUXCOM_data_annual
        im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds1,cmap=cmap1,vmin=vmn1,vmax =vmx1)
        ax.set(title='',xlabel='Longitude')
        ax.set_ylabel('Latitude', fontsize=15)
    elif i_col ==1:
        plot_data = Def_data_annaul - FLUXCOM_data_annual
        im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds2,cmap=cmap2,vmin=vmn2,vmax =vmx2)
        ax.set(title='', ylabel='',xlabel='Longitude')
    elif i_col ==2:
        plot_data = Mod1_data_annaul - FLUXCOM_data_annual
        im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds2,cmap=cmap2,vmin=vmn2,vmax =vmx2)
        ax.set(title='', ylabel='',xlabel='Longitude')
    else:
        plot_data = Mod2_data_annaul - FLUXCOM_data_annual
        im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds2,cmap=cmap2,vmin=vmn2,vmax =vmx2)
        ax.set(title='', ylabel='',xlabel='Longitude')
                
# if i_row<6:
# ax, _ = mpl.colorbar.make_axes(axes3.ravel().tolist())

# cbar = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, 
#                         spacing='proportional',ticks=bounds, boundaries=bounds, format='%1.1f')
# cbar.set_ticks(np.linspace(0,0.3,4))
# cbar.set_label('LH (kgC/m\u00b2/mon)')
# cbar1 = fig3.colorbar(im,ax=axes3[0],cmap=cmap1,shrink=0.5)
# cbar1.set_label('LH (kgC/m\u00b2/mon)')
# cbar1.set_ticks(np.linspace(vmn1,vmx1,4))

cbar2 = fig3.colorbar(im,ax=axes3,shrink=0.75)
cbar2.set_label('LH (kgC/m\u00b2/mon)')
cbar2.set_ticks(np.linspace(vmn2,vmx2,5))

# cbar1.remove()
# cbar2.remove()
# fig3.savefig(wrk_dir+'Monthly LH FLUXCOM_CLM5 over India (2000 to 2014)_Jul to Dec_3.png', 
#               dpi=600, bbox_inches="tight")
#%%Plotting FLUXCOM CLM5 SH MOnthly mean
########################################## 
fig2, axes2 = plt.subplots(nrows=6, ncols=4, sharex=True, sharey=True, dpi=600, layout='constrained')
fig2.set_size_inches(8,12)
# fig3.subplots_adjust(top=0.9)
# fig3.suptitle('Monthly LH over India (2000 to 2014)', y=1.0)
# fig3.tight_layout(pad=0.4)

vmn = 0
vmx = 150
N1 = 7
cmap = cmc.batlowW_r
# cmap = cmc.bamako_r  # define the colormap
# extract all colors from the .jet map
cmaplist = [cmap(i) for i in np.arange(0,cmap.N,int(256/N1))]
# create the new map
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist[1:], N1)
# define the bins and normalize
bounds = np.linspace(vmn, vmx, N1)
norm = mpl.colors.BoundaryNorm(bounds, N1)

# month_dim = [10, 11, 0, 1, 2]
# month_names = ['Nov','Dec','Jan','Feb','Mar']
for i_row in range(6):
    for i_col in range(4):
        p_row = i_row
        ax = axes2[p_row,i_col]
        ax.tick_params('both', labelsize=15)
        India.plot(facecolor='gray',edgecolor='black',ax=ax)
        if i_row<6 and i_row!=5:
            if i_col==0:
                plot_data = clip_SH_FLUXCOM[i_row,:,:]
                im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
                ax.set(title='',xlabel='')
                ax.set_ylabel(month[str(i_row+1)]+' \n\n Latitude', fontsize=15)
            elif i_col ==1:
                plot_data = clip_SH_Def[i_row,:,:]
                im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
                ax.set(title='', ylabel='',xlabel='')
            elif i_col ==2:
                plot_data = clip_SH_Mod1[i_row,:,:]
                im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
                ax.set(title='', ylabel='',xlabel='')
            else:
                plot_data = clip_SH_Mod2[i_row,:,:]
                im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
                ax.set(title='', ylabel='',xlabel='')   
        elif i_row<6 and i_row==5:
            if i_col==0:
                plot_data = clip_SH_FLUXCOM[i_row,:,:]
                im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
                ax.set(title='')
                ax.set_xlabel('Longitude', fontsize=15)  
                ax.set_ylabel(month[str(i_row+1)]+' \n\n Latitude', fontsize=15)
            elif i_col ==1:
                plot_data = clip_SH_Def[i_row,:,:]
                im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
                ax.set(title='', ylabel='')
                ax.set_xlabel('Longitude', fontsize=15)              
            elif i_col ==2:
                plot_data = clip_SH_Mod1[i_row,:,:]
                im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
                ax.set(title='', ylabel='')
                ax.set_xlabel('Longitude', fontsize=15)
            else:
                plot_data = clip_SH_Mod2[i_row,:,:]
                im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
                ax.set(title='',ylabel='')
                ax.set_xlabel('Longitude', fontsize = 15)                   
        elif i_row>=6 and i_row==11:
            if i_col==0:
                plot_data = clip_SH_FLUXCOM[i_row,:,:]
                im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
                ax.set(title='', ylabel='')
                ax.set_xlabel('Longitude', fontsize=15)
                ax.yaxis.set_tick_params(labelright=False)
                
            elif i_col ==1:
                plot_data = clip_SH_Def[i_row,:,:]
                im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
                ax.set(title='', ylabel='')
                ax.set_xlabel('Longitude', fontsize=15)
                
            elif i_col ==2:
                plot_data = clip_SH_Mod1[i_row,:,:]
                im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
                ax.set(title='', ylabel='')
                ax.set_xlabel('Longitude', fontsize=15)            
            else:
                plot_data = clip_SH_Mod2[i_row,:,:]
                im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
                ax.set(title='')
                ax.set_xlabel('Longitude', fontsize = 15)
                ax.set_ylabel('Latitude\n \n'+month[str(i_row+1)], fontsize=15)
                ax.yaxis.set_tick_params(labelright=True)
                
        elif i_row>=6 and i_row!=11:
            if i_col==0:
                plot_data = clip_SH_FLUXCOM[i_row,:,:]
                im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
                ax.set(title='', ylabel='',xlabel='')
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
                ax.yaxis.set_tick_params(labelright=False)
            elif i_col ==1:
                plot_data = clip_SH_Def[i_row,:,:]
                im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
                ax.set(title='', ylabel='',xlabel='')
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
            elif i_col ==2:
                plot_data = clip_SH_Mod1[i_row,:,:]
                im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
                ax.set(title='', ylabel='',xlabel='')
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
            else:
                plot_data = clip_SH_Mod2[i_row,:,:]
                im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
                ax.set(title='',xlabel='')
                ax.set_ylabel('Latitude\n \n'+month[str(i_row+1)], fontsize=15)
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
                ax.yaxis.set_tick_params(labelright=True)
# if i_row<6:
# ax, _ = mpl.colorbar.make_axes(axes2.ravel().tolist())
cbar = fig3.colorbar(im,ax=axes3[:,3],shrink=0.5)
cbar.set_label('SH (W/m\u00b2)')
cbar.set_ticks(np.linspace(vmn,vmx,4))

# if i_row>6:
#     cbar.remove()
# fig2.savefig(wrk_dir+'Monthly SH FLUXCOM_CLM5 over India (2000 to 2014)_Jan to Jul_2.png', 
#               dpi=600, bbox_inches="tight")
#%%
figcbar,axcbar = plt.subplots(nrows=3,ncols=4)

cbar = figcbar.colorbar(im,ax=axcbar[:,3],shrink=0.8)
cbar.set_label('LH (W/m\u00b2)')
cbar.set_ticks(np.linspace(vmn,vmx,4))
#%% Plotting LH in one sheet
fig3, axes3 = plt.subplots(nrows=6, ncols=8, sharex=True, sharey=True, dpi=600, layout='constrained')
fig3.set_size_inches(16,12)
# fig3.subplots_adjust(top=0.9)
# fig3.suptitle('Monthly LH over India (2000 to 2014)', y=1.0)
# fig3.tight_layout(pad=0.4)
cmap = cmc.batlowW_r  # define the colormap
# extract all colors from the .jet map
# cmaplist = [cmap(i) for i in range(cmap.N)]
cmaplist_2 = [cmap(i) for i in np.arange(0,cmap.N,int(256/13))]
# create the new map
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist_2, cmap.N)
# cmap[0] = [1,1,1,1.0]
# define the bins and normalize
bounds = np.linspace(0, 0.3, 13)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
# month_dim = [10, 11, 0, 1, 2]
# month_names = ['Nov','Dec','Jan','Feb','Mar']
for i_row in range(6):
    for i_col in range(8):
        p_row = i_row
        ax = axes3[p_row,i_col]
        India.plot(facecolor='gray',edgecolor='black',ax=ax)
        if i_col==0:
            plot_data = clip_LH_FLUXCOM[i_row,:,:]
            im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
            ax.set(title='', ylabel=month[str(i_row+1)]+' \n\n Latitude',xlabel='')
            ax.tick_params('both', labelsize=15)
        elif i_col ==7:
            plot_data = clip_LH_Def[i_row+6,:,:]
            im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
            ax.set(title='', ylabel='Latitude\n \n'+month[str(i_row+1)],xlabel='')
            ax.yaxis.set_tick_params(labelright=True)
        elif i_col>0 and i_col<5:
            plot_data = clip_LH_Mod1[i_row,:,:]
            im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
            ax.set(title='', ylabel='',xlabel='')
        else:
            plot_data = clip_LH_Mod2[i_row+6,:,:]
            im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
            ax.set(title='', ylabel='',xlabel='')
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()

# if i_row<6:
# ax, _ = mpl.colorbar.make_axes(axes3.ravel().tolist())

# cbar = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, 
#                         spacing='proportional',ticks=bounds, boundaries=bounds, format='%1.1f')
# cbar.set_ticks(np.linspace(0,0.3,4))
# cbar.set_label('LH (kgC/m\u00b2/mon)')
cbar = fig3.colorbar(im,ax=axes3[:,3],shrink=0.5)
cbar.set_label('LH (kgC/m\u00b2/mon)')
cbar.set_ticks(np.linspace(0,0.3,4))

# else:
#             # define the bins and normalize
#             bounds = np.linspace(0, 0.3, 13)
#             norm = mpl.colors.BoundaryNorm(bounds, cmap.N)


#             ax, _ = mpl.colorbar.make_axes(axes3.ravel().tolist())

#             cbar = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, location='left',
#                                     spacing='proportional',ticks=bounds, boundaries=bounds, format='%1.1f')
#             cbar.set_ticks(np.linspace(0,0.3,4))
#             cbar.set_label('(kgC/m\u00b2/mon)')

# if i_row>6:
cbar.remove()
# fig3.savefig(wrk_dir+'Monthly LH FLUXCOM_CLM5 over India (2000 to 2014)_Jan to Mar_2.png', 
#               dpi=600, bbox_inches="tight")

#%% Overall LH Annual sum

LH_FLUXCOM_Annual = np.sum(np.mean(np.mean(LH_FLUXCOM,axis=1),axis=1))
LH_Def_Annual = np.sum(np.mean(np.mean(LH_Def,axis=1),axis=1))
LH_Mod1_Annual = np.sum(np.mean(np.mean(LH_Mod1,axis=1),axis=1))
LH_Mod2_Annual = np.sum(np.mean(np.mean(LH_Mod2,axis=1),axis=1))

Names = ['FLUXCOM', 'Def', 'Mod1', 'Mod2']
Data = [LH_FLUXCOM_Annual, LH_Def_Annual, LH_Mod1_Annual, LH_Mod2_Annual]

fig, ax = plt.subplots()

ax.bar(Names, Data)

ax.set_ylabel('LH (kgC/m2)')
# ax.set_title('')
# ax.legend(title='Fruit color')

plt.show()

