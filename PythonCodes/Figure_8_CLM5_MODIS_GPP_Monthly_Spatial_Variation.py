#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 18:05:39 2024

@author: knreddy
"""

#%% Loading required Modules
import xarray as xr
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import matplotlib as mpl
import cmcrameri.cm as cmc
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
data_GPP = xr.open_dataset(data_dir+'CLM5_MODIS_GPPData_24-Feb-2024.nc')

India = gpd.read_file(shp_dir+'/india_administrative_outline_boundary.shp', crs="epsg:4326")

lat = data_GPP['lat']
lon = data_GPP['lon']

GPP_Def = data_GPP['GPP_Def']
GPP_Mod1 = data_GPP['GPP_Mod1']
GPP_Mod2 = data_GPP['GPP_Mod2']
GPP_MODIS = data_GPP['GPP_MODIS']
#%% Clipping Data for Indian region
GPP_Def.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
GPP_Def.rio.write_crs("epsg:4326", inplace=True)
clip_GPP_Def = GPP_Def.rio.clip(India.geometry, India.crs, drop=True)

GPP_Mod1.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
GPP_Mod1.rio.write_crs("epsg:4326", inplace=True)
clip_GPP_Mod1 = GPP_Mod1.rio.clip(India.geometry, India.crs, drop=True)

GPP_Mod2.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
GPP_Mod2.rio.write_crs("epsg:4326", inplace=True)
clip_GPP_Mod2 = GPP_Mod2.rio.clip(India.geometry, India.crs, drop=True)

GPP_MODIS.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
GPP_MODIS.rio.write_crs("epsg:4326", inplace=True)
clip_GPP_MODIS = GPP_MODIS.rio.clip(India.geometry, India.crs, drop=True)
#%% Plotting Monthly GPP: CLM5 vs MODIS
titles = ['MODIS', 'CLM5_Def', 'CLM5_Mod1', 'CLM5_Mod2']

fig3, axes3 = plt.subplots(nrows=6, ncols=4, sharex=True, sharey=True, dpi=600, layout='constrained')
fig3.set_size_inches(8,12)

cmap = cmc.batlowW_r  # define the colormap

cmaplist_2 = [cmap(i) for i in np.arange(0,cmap.N,int(256/13))]
# create the new map
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist_2[1:], 13)

# define the bins and normalize
bounds = np.linspace(0, 0.3, 13)
norm = mpl.colors.BoundaryNorm(bounds, 14)

for i_row in range(6,12):  
#for i_row in range(6): #for plotting Jan to Jun
    for i_col in range(4):
        p_row = i_row-6
        ax = axes3[p_row,i_col]
        ax.tick_params('both', labelsize=15)
        India.plot(facecolor='gray',edgecolor='black',ax=ax)
        if i_row<6 and i_row!=5:
            if i_col==0:
                plot_data = clip_GPP_MODIS[i_row,:,:]
                im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds,cmap=cmap,vmin=0,vmax =0.3)
                ax.set(title='',xlabel='')
                ax.set_ylabel(month[str(i_row+1)]+' \n\n Latitude', fontsize=15)
            elif i_col ==1:
                plot_data = clip_GPP_Def[i_row,:,:]
                im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds,cmap=cmap,vmin=0,vmax =0.3)
                ax.set(title='', ylabel='',xlabel='')
            elif i_col ==2:
                plot_data = clip_GPP_Mod1[i_row,:,:]
                im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds,cmap=cmap,vmin=0,vmax =0.3)
                ax.set(title='', ylabel='',xlabel='')
            else:
                plot_data = clip_GPP_Mod2[i_row,:,:]
                im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds,cmap=cmap,vmin=0,vmax =0.3)
                ax.set(title='', ylabel='',xlabel='')
        elif i_row<6 and i_row==5:
            if i_col==0:
                plot_data = clip_GPP_MODIS[i_row,:,:]
                im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds,cmap=cmap,vmin=0,vmax =0.3)
                ax.set(title='')
                ax.set_xlabel('Longitude', fontsize=15)  
                ax.set_ylabel(month[str(i_row+1)]+' \n\n Latitude', fontsize=15)
            elif i_col ==1:
                plot_data = clip_GPP_Def[i_row,:,:]
                im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds,cmap=cmap,vmin=0,vmax =0.3)
                ax.set(title='', ylabel='')
                ax.set_xlabel('Longitude', fontsize=15)              
            elif i_col ==2:
                plot_data = clip_GPP_Mod1[i_row,:,:]
                im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds,cmap=cmap,vmin=0,vmax =0.3)
                ax.set(title='', ylabel='')
                ax.set_xlabel('Longitude', fontsize=15)
            else:
                plot_data = clip_GPP_Mod2[i_row,:,:]
                im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds,cmap=cmap,vmin=0,vmax =0.3)
                ax.set(title='',ylabel='')
                ax.set_xlabel('Longitude', fontsize = 15)

        elif i_row>=6 and i_row==11:
            if i_col==0:
                plot_data = clip_GPP_MODIS[i_row,:,:]
                im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds,cmap=cmap,vmin=0,vmax =0.3)
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
                ax.set(title='', ylabel='')
                ax.set_xlabel('Longitude', fontsize=15)
                ax.yaxis.set_tick_params(labelright=False)
                
            elif i_col ==1:
                plot_data = clip_GPP_Def[i_row,:,:]
                im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds,cmap=cmap,vmin=0,vmax =0.3)
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
                ax.set(title='', ylabel='')
                ax.set_xlabel('Longitude', fontsize=15)
                
            elif i_col ==2:
                plot_data = clip_GPP_Mod1[i_row,:,:]
                im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds,cmap=cmap,vmin=0,vmax =0.3)
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
                ax.set(title='', ylabel='')
                ax.set_xlabel('Longitude', fontsize=15)
            else:
                plot_data = clip_GPP_Mod2[i_row,:,:]
                im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds,cmap=cmap,vmin=0,vmax =0.3)
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
                ax.set(title='')
                ax.set_xlabel('Longitude', fontsize = 15)
                ax.set_ylabel('Latitude\n \n'+month[str(i_row+1)], fontsize=15)
                ax.yaxis.set_tick_params(labelright=True)
                
        elif i_row>=6 and i_row!=11:
            if i_col==0:
                plot_data = clip_GPP_MODIS[i_row,:,:]
                im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds,cmap=cmap,vmin=0,vmax =0.3)
                ax.set(title='', ylabel='',xlabel='')
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
                ax.yaxis.set_tick_params(labelright=False)
            elif i_col ==1:
                plot_data = clip_GPP_Def[i_row,:,:]
                im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds,cmap=cmap,vmin=0,vmax =0.3)
                ax.set(title='', ylabel='',xlabel='')
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
            elif i_col ==2:
                plot_data = clip_GPP_Mod1[i_row,:,:]
                im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds,cmap=cmap,vmin=0,vmax =0.3)
                ax.set(title='', ylabel='',xlabel='')
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
            else:
                plot_data = clip_GPP_Mod2[i_row,:,:]
                im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds,cmap=cmap,vmin=0,vmax =0.3)
                ax.set(title='',xlabel='')
                ax.set_ylabel('Latitude\n \n'+month[str(i_row+1)], fontsize=15)
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
                ax.yaxis.set_tick_params(labelright=True)

cbar = fig3.colorbar(im,ax=axes3[:,3],shrink=0.5)
cbar.set_label('GPP (kgC/m\u00b2/mon)')
cbar.set_ticks(np.linspace(0,0.3,4))

cbar.remove()
if i_row>6:
    fig3.savefig(wrk_dir+'Monthly GPP MODIS_CLM5 over India (2000 to 2014)_Jan to Jun_'+str(date.today())+'.png', 
                  dpi=600, bbox_inches="tight") 
else:        
    fig3.savefig(wrk_dir+'Monthly GPP MODIS_CLM5 over India (2000 to 2014)_Jul to Dec_'+str(date.today())+'.png', 
                  dpi=600, bbox_inches="tight")
#%% Plotting Monthly GPP: CLM5 vs MODIS ///// comparing with Diff. plots
titles = ['MODIS', 'CLM5_Def', 'CLM5_Mod1', 'CLM5_Mod2']

fig3, axes3 = plt.subplots(nrows=6, ncols=4, sharex=True, sharey=True, dpi=600, layout='constrained')
fig3.set_size_inches(8,12)

cmap = cmc.batlowW_r  # define the colormap

cmaplist_2 = [cmap(i) for i in np.arange(0,cmap.N,int(256/13))]
# create the new map
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist_2[1:], 13)

N2 = 16
# create the new map
cmap2 = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', 
                                                    colors =[cmc.vik(40), 
                                                          (1, 1., 1), 
                                                          cmc.vik(220)], 
                                                    N=N2+1)
# cmap[0] = [1,1,1,1.0]
# define the bins and normalize
bounds = np.linspace(0, 0.3, 13)
norm = mpl.colors.BoundaryNorm(bounds, 14)

bounds2 = np.linspace(-0.2, 0.2, N2)
norm2 = mpl.colors.BoundaryNorm(bounds2,N2+1)

# for i_row in range(6,12): #for plotting Jul to dec maps
for i_row in range(6):
    for i_col in range(4):
        # p_row = i_row-6
        p_row = i_row
        ax = axes3[p_row,i_col]
        ax.tick_params('both', labelsize=15)
        India.plot(facecolor='gray',edgecolor='black',ax=ax)
        if i_row<6 and i_row!=5:
            if i_col==0:
                plot_data = clip_GPP_MODIS[i_row,:,:]
                im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds,cmap=cmap,vmin=0,vmax =0.3)
                ax.set(title='',xlabel='')
                ax.set_ylabel(month[str(i_row+1)]+' \n\n Latitude', fontsize=15)
            elif i_col ==1:
                plot_data = clip_GPP_Def[i_row,:,:] - clip_GPP_MODIS[i_row,:,:]
                im2 = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds2,cmap=cmap2,vmin=-0.2,vmax =0.2,extend='both')
                ax.set(title='', ylabel='',xlabel='')
            elif i_col ==2:
                plot_data = clip_GPP_Mod1[i_row,:,:] - clip_GPP_MODIS[i_row,:,:]
                im2 = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds2,cmap=cmap2,vmin=-0.2,vmax =0.2,extend='both')
                ax.set(title='', ylabel='',xlabel='')
            else:
                plot_data = clip_GPP_Mod2[i_row,:,:] - clip_GPP_MODIS[i_row,:,:]
                im2 = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds2,cmap=cmap2,vmin=-0.2,vmax =0.2,extend='both')
                ax.set(title='', ylabel='',xlabel='')
        elif i_row<6 and i_row==5:
            if i_col==0:
                plot_data = clip_GPP_MODIS[i_row,:,:]
                im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds,cmap=cmap,vmin=0,vmax =0.3)
                ax.set(title='')
                ax.set_xlabel('Longitude', fontsize=15)  
                ax.set_ylabel(month[str(i_row+1)]+' \n\n Latitude', fontsize=15)
            elif i_col ==1:
                plot_data = clip_GPP_Def[i_row,:,:] - clip_GPP_MODIS[i_row,:,:]
                im2 = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds2,cmap=cmap2,vmin=-0.2,vmax =0.2,extend='both')
                ax.set(title='', ylabel='')
                ax.set_xlabel('Longitude', fontsize=15)              
            elif i_col ==2:
                plot_data = clip_GPP_Mod1[i_row,:,:] - clip_GPP_MODIS[i_row,:,:]
                im2 = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds2,cmap=cmap2,vmin=-0.2,vmax =0.2,extend='both')
                ax.set(title='', ylabel='')
                ax.set_xlabel('Longitude', fontsize=15)
            else:
                plot_data = clip_GPP_Mod2[i_row,:,:] - clip_GPP_MODIS[i_row,:,:]
                im2 = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds2,cmap=cmap2,vmin=-0.2,vmax =0.2,extend='both')
                ax.set(title='',ylabel='')
                ax.set_xlabel('Longitude', fontsize = 15)

        elif i_row>=6 and i_row==11:
            if i_col==0:
                plot_data = clip_GPP_MODIS[i_row,:,:]
                im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds,cmap=cmap,vmin=0,vmax =0.3)
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
                ax.set(title='', ylabel='')
                ax.set_xlabel('Longitude', fontsize=15)
                ax.yaxis.set_tick_params(labelright=False)
                
            elif i_col ==1:
                plot_data = clip_GPP_Def[i_row,:,:] - clip_GPP_MODIS[i_row,:,:]
                im2 = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds2,cmap=cmap2,vmin=-0.2,vmax =0.2,extend='both')
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
                ax.set(title='', ylabel='')
                ax.set_xlabel('Longitude', fontsize=15)
                
            elif i_col ==2:
                plot_data = clip_GPP_Mod1[i_row,:,:] - clip_GPP_MODIS[i_row,:,:]
                im2 = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds2,cmap=cmap2,vmin=-0.2,vmax =0.2,extend='both')
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
                ax.set(title='', ylabel='')
                ax.set_xlabel('Longitude', fontsize=15)
            else:
                plot_data = clip_GPP_Mod2[i_row,:,:] - clip_GPP_MODIS[i_row,:,:]
                im2 = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds2,cmap=cmap2,vmin=-0.2,vmax =0.2,extend='both')
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
                ax.set(title='')
                ax.set_xlabel('Longitude', fontsize = 15)
                ax.set_ylabel('Latitude\n \n'+month[str(i_row+1)], fontsize=15)
                ax.yaxis.set_tick_params(labelright=True)
                
        elif i_row>=6 and i_row!=11:
            if i_col==0:
                plot_data = clip_GPP_MODIS[i_row,:,:]
                im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds,cmap=cmap,vmin=0,vmax =0.3)
                ax.set(title='', ylabel='',xlabel='')
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
                ax.yaxis.set_tick_params(labelright=False)
            elif i_col ==1:
                plot_data = clip_GPP_Def[i_row,:,:] - clip_GPP_MODIS[i_row,:,:]
                im2 = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds2,cmap=cmap2,vmin=-0.2,vmax =0.2,extend='both')
                ax.set(title='', ylabel='',xlabel='')
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
            elif i_col ==2:
                plot_data = clip_GPP_Mod1[i_row,:,:] - clip_GPP_MODIS[i_row,:,:]
                im2 = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds2,cmap=cmap2,vmin=-0.2,vmax =0.2,extend='both')
                ax.set(title='', ylabel='',xlabel='')
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
            else:
                plot_data = clip_GPP_Mod2[i_row,:,:] - clip_GPP_MODIS[i_row,:,:]
                im2 = ax.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds2,cmap=cmap2,vmin=-0.2,vmax =0.2,extend='both')
                ax.set(title='',xlabel='')
                ax.set_ylabel('Latitude\n \n'+month[str(i_row+1)], fontsize=15)
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
                ax.yaxis.set_tick_params(labelright=True)

cbar = fig3.colorbar(im,ax=axes3[:,3],shrink=0.5)
cbar.set_label('GPP (kgC/m\u00b2/mon)')
cbar.set_ticks(np.linspace(0,0.3,4))

cbar.remove()
if i_row>6:
    fig3.savefig(wrk_dir+'Monthly GPP MODIS_CLM5 over India (2000 to 2014)_Jan to Jun_'+str(date.today())+'.png', 
                  dpi=600, bbox_inches="tight") 
else:        
    fig3.savefig(wrk_dir+'Monthly GPP MODIS_CLM5 over India (2000 to 2014)_Jul to Dec_'+str(date.today())+'.png', 
                  dpi=600, bbox_inches="tight")
#%%
figcbar,axcbar = plt.subplots(nrows=3,ncols=4)

cbar = figcbar.colorbar(im,ax=axcbar[:,3],shrink=0.8)
cbar.set_label('GPP (kgC/m\u00b2/mon)')
cbar.set_ticks(np.linspace(0,0.3,4))

figcbar2,axcbar2 = plt.subplots(nrows=3,ncols=4)

cbar2 = figcbar2.colorbar(im2,ax=axcbar2[:,3],shrink=0.8)
cbar2.set_label('GPP (kgC/m\u00b2/mon)')
cbar2.set_ticks(np.linspace(-0.2,0.2,5))