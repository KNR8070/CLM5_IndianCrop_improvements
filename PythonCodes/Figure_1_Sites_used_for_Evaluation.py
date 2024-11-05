#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 13:26:35 2024

@author: knreddy
"""
#%% Import libraries
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from cmcrameri import cm
import matplotlib as mpl
from scipy.stats import rankdata
import string
import xarray as xr
from datetime import date
#%% Loading data and creating wheat geodataframe
data_dir = '/Users/knreddy/Documents/GMD_Paper/Data_for_plotting_Figures/'

Rice_GY_df = pd.read_excel(data_dir+'SiteScale_Data/Rice_GY.xlsx',index_col=0)
Wheat_LAI_df = pd.read_excel(data_dir+'SiteScale_Data/Wheat_LAI.xlsx',index_col=0)
#%% Loading crop maps data
India = gpd.read_file(data_dir+'/India_Shapefile_/india_administrative_outline_boundary.shp', crs="epsg:4326")
data = xr.open_dataset(data_dir+'CLM5_PCT_CFT_SW_Rice.nc')

CFT_SW_Rainfed = data['PCT_SW_RF']
CFT_SW_Irrigated = data['PCT_SW_IR']
CFT_Ri_Rainfed = data['PCT_Ri_RF']
CFT_Ri_Irrigated = data['PCT_Ri_IR']

lat = data['lat']
lon = data['lon']
year = data['year']

CFT_SW_Rainfed_2000_14 = np.mean(CFT_SW_Rainfed[year>1999,:,:],axis=0)
CFT_SW_Irrigated_2000_14 = np.mean(CFT_SW_Irrigated[year>1999,:,:],axis=0)
CFT_SW_2000_14 = CFT_SW_Rainfed_2000_14 + CFT_SW_Irrigated_2000_14
CFT_SW_2000_14.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
CFT_SW_2000_14.rio.write_crs("epsg:4326", inplace=True)
clipped_CFT_SW_2000_14 = CFT_SW_2000_14.rio.clip(India.geometry, India.crs, drop=True)

CFT_Ri_Rainfed_2000_14 = np.mean(CFT_Ri_Rainfed[year>1999,:,:],axis=0)
CFT_Ri_Irrigated_2000_14 = np.mean(CFT_Ri_Irrigated[year>1999,:,:],axis=0)
CFT_Ri_2000_14 = CFT_Ri_Rainfed_2000_14 + CFT_Ri_Irrigated_2000_14
CFT_Ri_2000_14.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
CFT_Ri_2000_14.rio.write_crs("epsg:4326", inplace=True)
clipped_CFT_Ri_2000_14 = CFT_Ri_2000_14.rio.clip(India.geometry, India.crs, drop=True)
#%% plotting
fig,ax= plt.subplots(ncols=2,sharey=True,layout='constrained',figsize=(7,4))
fsize=12

cmap = cm.grayC_r
cmaplist_2 = [cmap(i) for i in np.arange(0,cmap.N,int(256/41))]
# create the new map
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist_2[3:40], 11)
# define the bins and normalize
bounds = np.linspace(0, 100, 11)
norm = mpl.colors.BoundaryNorm(bounds, 12)

arrowprops_kw=dict(facecolor='black', arrowstyle='->', lw=1)
bbox_kw=dict(boxstyle="square",fc="white",lw=0.1)

crop=['Wheat','Rice']
var=['LAI','GY']
crop_m = ['SW','Ri']

for i_c_n,i_crop in enumerate(crop):
        axes = ax[i_c_n]
        India.plot(ax = axes,facecolor='none',edgecolor='black')
        crop_map = 'clipped_CFT_'+crop_m[i_c_n]+'_2000_14'
        plot_data = eval(crop_map)
        im = axes.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds,cmap=cmap)        
        data = i_crop+'_'+var[i_c_n]+'_gfd'
        
        for i_num,i_site in enumerate(eval(data).Site):
            axes.scatter(eval(data)["Lon"][i_num],eval(data)["Lat"][i_num],marker = 'o',
                               s=50,c='red')
        
        if i_crop == 'Wheat':            
            for i_num, i_site in enumerate(eval(data)['Site']):
                if i_site == 'Jobner':
                    axes.annotate(i_site,xy=(eval(data)['Lon'][i_num],eval(data)['Lat'][i_num]),
                                xytext=(eval(data)['Lon'][i_num]-8,eval(data)['Lat'][i_num]+2),
                                arrowprops=arrowprops_kw, bbox=bbox_kw,fontsize=fsize-4)
                elif i_site == 'Faizabad' :
                    axes.annotate(i_site,xy=(eval(data)['Lon'][i_num],eval(data)['Lat'][i_num]),
                                xytext=(eval(data)['Lon'][i_num]-12,eval(data)['Lat'][i_num]-2), 
                                arrowprops=arrowprops_kw, bbox=bbox_kw,fontsize=fsize-4)
                elif (i_site == 'Meerut') :
                    axes.annotate(i_site,xy=(eval(data)['Lon'][i_num],eval(data)['Lat'][i_num]),
                                xytext=(eval(data)['Lon'][i_num]+1,eval(data)['Lat'][i_num]+5), 
                                arrowprops=arrowprops_kw, bbox=bbox_kw,fontsize=fsize-4)
                elif (i_site == 'Ludhiana') :
                    axes.annotate(i_site,xy=(eval(data)['Lon'][i_num],eval(data)['Lat'][i_num]),
                                xytext=(eval(data)['Lon'][i_num]-8,eval(data)['Lat'][i_num]+2), 
                                arrowprops=arrowprops_kw,bbox=bbox_kw,fontsize=fsize-4)
                elif (i_site == 'Cooch_Behar'):
                    axes.annotate('Cooch Behar',xy=(eval(data)['Lon'][i_num],eval(data)['Lat'][i_num]),
                                xytext=(eval(data)['Lon'][i_num]-1,eval(data)['Lat'][i_num]+3), 
                                arrowprops=arrowprops_kw, bbox=bbox_kw,fontsize=fsize-4)
                elif (i_site == 'Pantnagar'):
                    axes.annotate(i_site,xy=(eval(data)['Lon'][i_num],eval(data)['Lat'][i_num]),
                                    xytext=(eval(data)['Lon'][i_num]+2,eval(data)['Lat'][i_num]+3), 
                                    arrowprops=arrowprops_kw, bbox=bbox_kw,fontsize=fsize-4)
                else:
                    axes.annotate(i_site,xy=(eval(data)['Lon'][i_num],eval(data)['Lat'][i_num]),
                                xytext=(eval(data)['Lon'][i_num]+4,eval(data)['Lat'][i_num]-2), 
                                arrowprops=arrowprops_kw, bbox=bbox_kw,fontsize=fsize-4)
        if i_crop == 'Rice':
            for i_num, i_site in enumerate(eval(data)['Site']):
                if i_site == 'Kuthulia':
                    axes.annotate(i_site,xy=(eval(data)['Lon'][i_num],eval(data)['Lat'][i_num]),
                                xytext=(eval(data)['Lon'][i_num]-10,eval(data)['Lat'][i_num]+2),
                                arrowprops=arrowprops_kw, bbox=bbox_kw,fontsize=fsize-4)
                elif ((i_site == 'Anathapur') or (i_site == 'Hyderabad')):
                    axes.annotate(i_site,xy=(eval(data)['Lon'][i_num],eval(data)['Lat'][i_num]),
                                xytext=(eval(data)['Lon'][i_num]+3,eval(data)['Lat'][i_num]), 
                                arrowprops=arrowprops_kw, bbox=bbox_kw,fontsize=fsize-4)
                elif (i_site == 'Kaul'):
                    axes.annotate(i_site,xy=(eval(data)['Lon'][i_num],eval(data)['Lat'][i_num]),
                                xytext=(eval(data)['Lon'][i_num]-1.2,eval(data)['Lat'][i_num]+4), 
                                arrowprops=arrowprops_kw, bbox=bbox_kw,fontsize=fsize-4)
                elif (i_site == 'Raipur'):
                    axes.annotate(i_site,xy=(eval(data)['Lon'][i_num],eval(data)['Lat'][i_num]),
                                xytext=(eval(data)['Lon'][i_num]-10,eval(data)['Lat'][i_num]-2), 
                                arrowprops=arrowprops_kw, bbox=bbox_kw,fontsize=fsize-4)
                else:
                    axes.annotate(i_site,xy=(eval(data)['Lon'][i_num],eval(data)['Lat'][i_num]),
                                xytext=(eval(data)['Lon'][i_num]+4,eval(data)['Lat'][i_num]), 
                                arrowprops=arrowprops_kw, bbox=bbox_kw,fontsize=fsize-4)               

ax[0].set_xticks(np.arange(70,105,10))
ax[1].set_xticks(np.arange(70,105,10))
# ax[1,0].set_xticklabels(np.arange(70,105,10),fontsize=fsize)
ax[0].set_yticks(np.arange(5,40,10))

# ax[1,0].set_yticklabels(np.arange(5,40,10),fontsize=fsize)
ax[0].set_xlabel('Longitude [\u00b0E]',fontsize=fsize)
ax[1].set_xlabel('Longitude [\u00b0E]',fontsize=fsize)
ax[0].set_ylabel('Latitude [\u00b0N]',fontsize=fsize)
ax[0].set_title('1. Wheat',fontsize=fsize+4)
ax[1].set_title('2. Rice',fontsize=fsize+4)

cbar = fig.colorbar(im,ax=ax[1],shrink=0.5,fraction=0.04,pad=0.001)
cbar.set_label('Crop Area (% of grid cell)',fontsize=fsize-2)
cbar.set_ticks(np.linspace(0,100,6))
cbar.set_ticklabels(np.linspace(0,100,6),fontsize=fsize-2)

savefig_dir = '/Users/knreddy/Documents/GMD_Paper/Figures/'
fig.savefig(savefig_dir+'Figure_1_Sites_used_for_evaluation_'+str(date.today())+'.png', 
              dpi=600, bbox_inches="tight")