#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 04 17:48:41 2024

@author: knreddy
"""
#%% Load Modules
import xarray as xr
import matplotlib.pyplot as plt
import geopandas
import numpy as np
import matplotlib as mpl
from cmcrameri import cm
from datetime import date
import string
#%% Loading datasets
data_dir = '/Users/knreddy/Documents/GMD_Paper/Data_for_Plotting_Figures/'
shp_dir = data_dir+'India_Shapefile_'

data = xr.open_dataset(data_dir+'CLM5_PCT_CFT_SW_Rice.nc')

India = geopandas.read_file(shp_dir+'/india_administrative_outline_boundary.shp', crs="epsg:4326")
#%% Extracting crop functional type percent crop areas
CFT_SW_Rainfed = data['PCT_SW_RF']
CFT_SW_Irrigated = data['PCT_SW_IR']
CFT_Ri_Rainfed = data['PCT_Ri_RF']
CFT_Ri_Irrigated = data['PCT_Ri_IR']

lat = data['lat']
lon = data['lon']
year = data['year']
#%% extrcating for 2000 to 2014 and Indian region
CFT_SW_Rainfed_2000_14 = np.mean(CFT_SW_Rainfed[year>1999,:,:],axis=0)
CFT_SW_Rainfed_2000_14.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
CFT_SW_Rainfed_2000_14.rio.write_crs("epsg:4326", inplace=True)
clipped_CFT_SW_Rainfed_2000_14 = CFT_SW_Rainfed_2000_14.rio.clip(India.geometry, India.crs, drop=True)

CFT_SW_Irrigated_2000_14 = np.mean(CFT_SW_Irrigated[year>1999,:,:],axis=0)
CFT_SW_Irrigated_2000_14.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
CFT_SW_Irrigated_2000_14.rio.write_crs("epsg:4326", inplace=True)
clipped_CFT_SW_Irrigated_2000_14 = CFT_SW_Irrigated_2000_14.rio.clip(India.geometry, India.crs, drop=True)

CFT_Ri_Rainfed_2000_14 = np.mean(CFT_Ri_Rainfed[year>1999,:,:],axis=0)
CFT_Ri_Rainfed_2000_14.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
CFT_Ri_Rainfed_2000_14.rio.write_crs("epsg:4326", inplace=True)
clipped_CFT_Ri_Rainfed_2000_14 = CFT_Ri_Rainfed_2000_14.rio.clip(India.geometry, India.crs, drop=True)

CFT_Ri_Irrigated_2000_14 = np.mean(CFT_Ri_Irrigated[year>1999,:,:],axis=0)
CFT_Ri_Irrigated_2000_14.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
CFT_Ri_Irrigated_2000_14.rio.write_crs("epsg:4326", inplace=True)
clipped_CFT_Ri_Irrigated_2000_14 = CFT_Ri_Irrigated_2000_14.rio.clip(India.geometry, India.crs, drop=True)
#%%################# Plotting Spatial plots
fig2, axes2 = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, dpi=600, layout='constrained',figsize=(6,5.75))
fig2.suptitle('CLM5 Crop Areas (Mean of 2000 to 2014)')

plotid_x = 0.02
plotid_y = 0.9

cmap = cm.batlowW_r
cmaplist_2 = [cmap(i) for i in np.arange(0,cmap.N,int(256/41))]
# create the new map
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist_2[1:], 41)
# define the bins and normalize
bounds = np.linspace(0, 100, 41)
norm = mpl.colors.BoundaryNorm(bounds, 42)

India.plot(facecolor='none',edgecolor='black',ax=axes2[0,0])
plot_data = clipped_CFT_SW_Rainfed_2000_14
axes2[0,0].contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds,cmap=cmap,vmin=0,vmax =100)
axes2[0,0].set(title = 'Rainfed', ylabel='Wheat \n \n Latitude',xlabel='')

India.plot(facecolor='none',edgecolor='black',ax=axes2[0,1])
plot_data = clipped_CFT_SW_Irrigated_2000_14
axes2[0,1].contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds,cmap=cmap,vmin=0,vmax =100)
axes2[0,1].set(title = 'Irrigated',ylabel='',xlabel='')

India.plot(facecolor='none',edgecolor='black',ax=axes2[1,0])
plot_data = clipped_CFT_Ri_Rainfed_2000_14
axes2[1,0].contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds,cmap=cmap,vmin=0,vmax =100)
axes2[1,0].set(ylabel='Rice \n \n Latitude',xlabel='Longitude')

India.plot(facecolor='none',edgecolor='black',ax=axes2[1,1])
plot_data = clipped_CFT_Ri_Irrigated_2000_14
axes2[1,1].contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds,cmap=cmap,vmin=0,vmax =100)
axes2[1,1].set(ylabel='',xlabel='Longitude')

for i_row in np.arange(2):
    for i_col in np.arange(2):
        ax = axes2[i_row,i_col]
        panel_no = list(string.ascii_lowercase)[i_row]+'.'+str(i_col+1)
        ax.text(plotid_x,plotid_y, panel_no,fontsize=10,transform=ax.transAxes)

# cax, kw = mpl.colorbar.make_axes([ax=axes2.flat])
# fig2.colorbar(im,cax=cax,**kw)
ax2, _ = mpl.colorbar.make_axes(axes2.ravel().tolist())

cbar = mpl.colorbar.ColorbarBase(ax2, cmap=cmap)
cbar.set_ticks(np.linspace(0,1,11))
cbar.set_label('Crop Area (% of grid cell)*100')

savefig_dir = '/Users/knreddy/Documents/GMD_Paper/Figures/'
fig2.savefig(savefig_dir+'CLM5_CropArea_'+str(date.today())+'.png', 
              dpi=600, bbox_inches="tight")
