#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 18:21:11 2024

@author: knreddy
"""

#%% Loading required Modules
import xarray as xr
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import matplotlib as mpl
import cmcrameri.cm as cmc
import string
from datetime import date
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
data_dir = '/Users/knreddy/Documents/GMD_Paper/Data_for_Plotting_Figures/'
shp_dir = data_dir+'India_Shapefile_'
    
month = {'1':'Jan','2':'Feb','3':'Mar','4':'Apr','5':'May','6':'Jun',
         '7':'Jul','8':'Aug','9':'Sep','10':'Oct','11':'Nov','12':'Dec'}
#%% Reading data for plotting
data_Yield = xr.open_dataset(data_dir+'CLM5_Earthstat_Yield_2005Data_25-Feb-2024.nc')

India = gpd.read_file(shp_dir+'/india_administrative_outline_boundary.shp', crs="epsg:4326")

lat = data_Yield['lat']
lon = data_Yield['lon']

Yield_SW_Def = data_Yield['Yield_SW_Def']
Yield_SW_Mod1 = data_Yield['Yield_SW_Mod1']
Yield_SW_Mod2 = data_Yield['Yield_SW_Mod2']
Yield_SW_Earthstat = data_Yield['Yield_SW_EarthStat']

Yield_Rice_Def = data_Yield['Yield_Rice_Def']
Yield_Rice_Mod1 = data_Yield['Yield_Rice_Mod1']
Yield_Rice_Mod2 = data_Yield['Yield_Rice_Mod2']
Yield_Rice_Earthstat = data_Yield['Yield_Rice_EarthStat']
#%% Clipping Data for Indian region
Yield_SW_Def.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
Yield_SW_Def.rio.write_crs("epsg:4326", inplace=True)
clip_Yield_SW_Def = Yield_SW_Def.rio.clip(India.geometry, India.crs, drop=True)

Yield_SW_Mod1.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
Yield_SW_Mod1.rio.write_crs("epsg:4326", inplace=True)
clip_Yield_SW_Mod1 = Yield_SW_Mod1.rio.clip(India.geometry, India.crs, drop=True)

Yield_SW_Mod2.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
Yield_SW_Mod2.rio.write_crs("epsg:4326", inplace=True)
clip_Yield_SW_Mod2 = Yield_SW_Mod2.rio.clip(India.geometry, India.crs, drop=True)

Yield_SW_Earthstat.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
Yield_SW_Earthstat.rio.write_crs("epsg:4326", inplace=True)
clip_Yield_SW_Earthstat = Yield_SW_Earthstat.rio.clip(India.geometry, India.crs, drop=True)

Yield_Rice_Def.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
Yield_Rice_Def.rio.write_crs("epsg:4326", inplace=True)
clip_Yield_Rice_Def = Yield_Rice_Def.rio.clip(India.geometry, India.crs, drop=True)

Yield_Rice_Mod1.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
Yield_Rice_Mod1.rio.write_crs("epsg:4326", inplace=True)
clip_Yield_Rice_Mod1 = Yield_Rice_Mod1.rio.clip(India.geometry, India.crs, drop=True)

Yield_Rice_Mod2.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
Yield_Rice_Mod2.rio.write_crs("epsg:4326", inplace=True)
clip_Yield_Rice_Mod2 = Yield_Rice_Mod2.rio.clip(India.geometry, India.crs, drop=True)

Yield_Rice_Earthstat.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
Yield_Rice_Earthstat.rio.write_crs("epsg:4326", inplace=True)
clip_Yield_Rice_Earthstat = Yield_Rice_Earthstat.rio.clip(India.geometry, India.crs, drop=True)
#%% Yield EarthStat and CLM5
Yield_Rice_Def_Diff = (clip_Yield_Rice_Def*30/1000) - clip_Yield_Rice_Earthstat
Yield_Rice_Mod1_Diff = (clip_Yield_Rice_Mod1*30/1000) - clip_Yield_Rice_Earthstat
Yield_Rice_Mod2_Diff = (clip_Yield_Rice_Mod2*30/1000) - clip_Yield_Rice_Earthstat

Yield_SW_Def_Diff = (clip_Yield_SW_Def*30/1000) - clip_Yield_SW_Earthstat
Yield_SW_Mod1_Diff = (clip_Yield_SW_Mod1*30/1000) - clip_Yield_SW_Earthstat
Yield_SW_Mod2_Diff = (clip_Yield_SW_Mod2*30/1000) - clip_Yield_SW_Earthstat

fig, axes = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=True, dpi=600, layout='constrained')
fig.set_size_inches(8,4)

plotid_x = 0.02
plotid_y = 0.9

cmap1 = cmc.lajolla_r  # define the colormap
# extract all colors from the .jet map
c1min = 0
c1max = 7
N1 = 15#no of color bins
cmaplist1 = [cmap1(i) for i in np.arange(0,cmap1.N,int(256/N1))]
# create the new map
cmap1 = mpl.colors.LinearSegmentedColormap.from_list(
    'Customcmap1', cmaplist1, N1)

# cmap2 = cmc.vik  # define the colormap
N2 = 16
# create the new map
cmap2 = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', 
                                                    colors =[cmc.vik(40), 
                                                          (1, 1., 1), 
                                                          cmc.vik(220)], 
                                                    N=N2+1)

fsize=8
# define the bins
bounds1 = np.linspace(0, 8, N1)
norm1 = mpl.colors.BoundaryNorm(bounds1,N1+1)

bounds2 = np.linspace(-4, 4, N2)
norm2 = mpl.colors.BoundaryNorm(bounds2,N2+1)

for i_col in range(4):
    for i_row in range(2):
        p_row = i_row
        ax = axes[p_row,i_col]
        India.plot(facecolor='gray',edgecolor='black',ax=ax)
        panel_no = list(string.ascii_lowercase)[i_row]+'.'+str(i_col+1)
        ax.text(plotid_x,plotid_y, panel_no,fontsize=fsize,transform=ax.transAxes)
        if i_row==0:
            if i_col==0:
                plot_data = Yield_SW_Earthstat
                im0 = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds1,norm=norm1,
                                  cmap=cmap1,vmin=0,vmax =8,extend='max')
                ax.set_title('EarthStat',fontsize=fsize)
                ax.set_ylabel('Wheat \n\nLatitude')
                ax.set_xlabel('')
                ax.set_xlim(67.5, 98)
                ax.set_ylim(5.5, 38)
            elif i_col ==1:
                plot_data = Yield_SW_Def_Diff
                im1 = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds2,
                                  cmap=cmap2,vmin=-4,vmax =4,extend='both')
                ax.set_title('CLM5_Def - EarthStat',fontsize=fsize)
                ax.set_ylabel('')
                ax.set_xlabel('')
            elif i_col ==2:
                plot_data = Yield_SW_Mod1_Diff
                im2 = ax.contourf(plot_data.lon,plot_data.lat,plot_data,
                                  levels=bounds2,cmap=cmap2,vmin=-4,vmax =4,extend='both')
                ax.set_title('CLM5_Mod1 - EarthStat',fontsize=fsize)
                ax.set_ylabel('')
                ax.set_xlabel('')
            else:
                plot_data = Yield_SW_Mod2_Diff
                im3 = ax.contourf(plot_data.lon,plot_data.lat,plot_data,
                                  levels=bounds2,cmap=cmap2,vmin=-4,vmax =4,extend='both')
                ax.set_title('CLM5_Mod2 - EarthStat',fontsize=fsize)
                ax.set_ylabel('')
                ax.set_xlabel('')
        else:
            if i_col==0:
                plot_data = Yield_Rice_Earthstat
                im4 = ax.contourf(plot_data.lon,plot_data.lat,plot_data,
                                  levels=bounds1,cmap=cmap1,vmin=0,vmax =8,extend='max')
                ax.set(title='', ylabel='Rice \n\nLatitude',xlabel='Longitude')
            elif i_col ==1:
                plot_data = Yield_Rice_Def_Diff
                im5 = ax.contourf(plot_data.lon,plot_data.lat,plot_data,
                                  levels=bounds2,cmap=cmap2,vmin=-4,vmax =4,extend='both')
                ax.yaxis.set_label_position("right")
                ax.set(title='', ylabel='',xlabel='Longitude')     
            elif i_col ==2:
                plot_data = Yield_Rice_Mod1_Diff
                im6 = ax.contourf(plot_data.lon,plot_data.lat,plot_data,
                                  levels=bounds2,cmap=cmap2,vmin=-4,vmax =4,extend='both')
                ax.yaxis.set_label_position("right")
                ax.set(title='', ylabel='',xlabel='Longitude') 
            else:
                plot_data = Yield_Rice_Mod2_Diff
                im7 = ax.contourf(plot_data.lon,plot_data.lat,plot_data,
                                  levels=bounds2,cmap=cmap2,vmin=-4,vmax =4,extend='both')
                ax.set(title='', ylabel='',xlabel='Longitude')
            
cbar1 = fig.colorbar(im4,ax=axes[:,0],shrink=0.5)
cbar1.set_label('Yield (t/ha)')
cbar1.set_ticks(np.linspace(0,8,9))

cbar2 = fig.colorbar(im7, ax=axes[:,3], shrink=0.5)
cbar2.set_label('Yield_difference (t/ha)')
cbar2.set_ticks(np.linspace(-4,4,5))

fig.savefig(wrk_dir+'Earthstat_CLM5_Yield_comparison_'+str(date.today())+'.png',dpi=600, bbox_inches="tight")