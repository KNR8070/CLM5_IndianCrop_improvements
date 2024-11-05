#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 00:56:40 2024

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
#%% Plotting Monthly Difference LH: CLM5 vs FLUXCOM
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
#%%Plotting FLUXCOM CLM5 SH MOnthly mean
########################################## 
fig2, axes2 = plt.subplots(nrows=6, ncols=4, sharex=True, sharey=True, dpi=600, layout='constrained')
fig2.set_size_inches(8,12)

vmn = 0
vmx = 150
N1 = 11

vmn2 = -75
vmx2 = 75
N2 = 16

#### cmap for FLUXCOM data
cmap = cmc.batlowW_r  
cmaplist = [cmap(i) for i in np.arange(0,cmap.N,int(256/N1))]
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist[1:], N1)
bounds = np.linspace(vmn, vmx, N1)
norm = mpl.colors.BoundaryNorm(bounds, N1+1)
#### cmap for CLM5-FLUXCOM data
cmap2 = cmc.vik  
cmap2 = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', 
                                                    colors =[cmc.vik(40), 
                                                          (1, 1., 1), 
                                                          cmc.vik(220)], 
                                                    N=N2+1)
bounds2 = np.linspace(vmn2, vmx2, N2)
norm2 = mpl.colors.BoundaryNorm(bounds2, N2+1)
for i_row in range(6,12):
# for i_row in range(6):
    for i_col in range(4):
        # p_row = i_row
        p_row = i_row-6
        ax = axes2[p_row,i_col]
        ax.tick_params('both', labelsize=15)
        India.plot(facecolor='gray',edgecolor='black',ax=ax)
        if i_row<6 and i_row!=5:
            if i_col==0:
                plot_data = clip_SH_FLUXCOM[i_row,:,:]
                im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,
                                 levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx,extend='both')
                ax.set(title='',xlabel='')
                ax.set_ylabel(month[str(i_row+1)]+' \n\n Latitude', fontsize=15)
            elif i_col ==1:
                plot_data = clip_SH_Def[i_row,:,:]-clip_SH_FLUXCOM[i_row,:,:]
                im2 = ax.contourf(plot_data.lon,plot_data.lat,plot_data,
                                 levels=bounds2,cmap=cmap2,vmin=vmn2,vmax=vmx2,extend='both')
                ax.set(title='', ylabel='',xlabel='')
            elif i_col ==2:
                plot_data = clip_SH_Mod1[i_row,:,:]-clip_SH_FLUXCOM[i_row,:,:]
                im2 = ax.contourf(plot_data.lon,plot_data.lat,plot_data,
                                 levels=bounds2,cmap=cmap2,vmin=vmn2,vmax=vmx2,extend='both')
                ax.set(title='', ylabel='',xlabel='')
            else:
                plot_data = clip_SH_Mod2[i_row,:,:]-clip_SH_FLUXCOM[i_row,:,:]
                im2 = ax.contourf(plot_data.lon,plot_data.lat,plot_data,
                                 levels=bounds2,cmap=cmap2,vmin=vmn2,vmax=vmx2,extend='both')
                ax.set(title='', ylabel='',xlabel='')   
        elif i_row<6 and i_row==5:
            if i_col==0:
                plot_data = clip_SH_FLUXCOM[i_row,:,:]
                im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,
                                 levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx,extend='both')
                ax.set(title='')
                ax.set_xlabel('Longitude', fontsize=15)  
                ax.set_ylabel(month[str(i_row+1)]+' \n\n Latitude', fontsize=15)
            elif i_col ==1:
                plot_data = clip_SH_Def[i_row,:,:]-clip_SH_FLUXCOM[i_row,:,:]
                im2 = ax.contourf(plot_data.lon,plot_data.lat,plot_data,
                                 levels=bounds2,cmap=cmap2,vmin=vmn2,vmax=vmx2,extend='both')
                ax.set(title='', ylabel='')
                ax.set_xlabel('Longitude', fontsize=15)              
            elif i_col ==2:
                plot_data = clip_SH_Mod1[i_row,:,:]-clip_SH_FLUXCOM[i_row,:,:]
                im2 = ax.contourf(plot_data.lon,plot_data.lat,plot_data,
                                 levels=bounds2,cmap=cmap2,vmin=vmn2,vmax=vmx2,extend='both')
                ax.set(title='', ylabel='')
                ax.set_xlabel('Longitude', fontsize=15)
            else:
                plot_data = clip_SH_Mod2[i_row,:,:]-clip_SH_FLUXCOM[i_row,:,:]
                im2 = ax.contourf(plot_data.lon,plot_data.lat,plot_data,
                                 levels=bounds2,cmap=cmap2,vmin=vmn2,vmax=vmx2,extend='both')
                ax.set(title='',ylabel='')
                ax.set_xlabel('Longitude', fontsize = 15)                   
        elif i_row>=6 and i_row==11:
            if i_col==0:
                plot_data = clip_SH_FLUXCOM[i_row,:,:]
                im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,
                                 levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx,extend='both')
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
                ax.set(title='', ylabel='')
                ax.set_xlabel('Longitude', fontsize=15)
                ax.yaxis.set_tick_params(labelright=False)
                
            elif i_col ==1:
                plot_data = clip_SH_Def[i_row,:,:]-clip_SH_FLUXCOM[i_row,:,:]
                im2 = ax.contourf(plot_data.lon,plot_data.lat,plot_data,
                                 levels=bounds2,cmap=cmap2,vmin=vmn2,vmax=vmx2,extend='both')
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
                ax.set(title='', ylabel='')
                ax.set_xlabel('Longitude', fontsize=15)
                
            elif i_col ==2:
                plot_data = clip_SH_Mod1[i_row,:,:]-clip_SH_FLUXCOM[i_row,:,:]
                im2 = ax.contourf(plot_data.lon,plot_data.lat,plot_data,
                                 levels=bounds2,cmap=cmap2,vmin=vmn2,vmax=vmx2,extend='both')
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
                ax.set(title='', ylabel='')
                ax.set_xlabel('Longitude', fontsize=15)            
            else:
                plot_data = clip_SH_Mod2[i_row,:,:]-clip_SH_FLUXCOM[i_row,:,:]
                im2 = ax.contourf(plot_data.lon,plot_data.lat,plot_data,
                                 levels=bounds2,cmap=cmap2,vmin=vmn2,vmax=vmx2,extend='both')
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
                ax.set(title='')
                ax.set_xlabel('Longitude', fontsize = 15)
                ax.set_ylabel('Latitude\n \n'+month[str(i_row+1)], fontsize=15)
                ax.yaxis.set_tick_params(labelright=True)
                
        elif i_row>=6 and i_row!=11:
            if i_col==0:
                plot_data = clip_SH_FLUXCOM[i_row,:,:]
                im = ax.contourf(plot_data.lon,plot_data.lat,plot_data,
                                 levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
                ax.set(title='', ylabel='',xlabel='')
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
                ax.yaxis.set_tick_params(labelright=False)
            elif i_col ==1:
                plot_data = clip_SH_Def[i_row,:,:]-clip_SH_FLUXCOM[i_row,:,:]
                im2 = ax.contourf(plot_data.lon,plot_data.lat,plot_data,
                                 levels=bounds2,cmap=cmap2,vmin=vmn2,vmax=vmx2,extend='both')
                ax.set(title='', ylabel='',xlabel='')
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
            elif i_col ==2:
                plot_data = clip_SH_Mod1[i_row,:,:]-clip_SH_FLUXCOM[i_row,:,:]
                im2 = ax.contourf(plot_data.lon,plot_data.lat,plot_data,
                                 levels=bounds2,cmap=cmap2,vmin=vmn2,vmax=vmx2,extend='both')
                ax.set(title='', ylabel='',xlabel='')
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
            else:
                plot_data = clip_SH_Mod2[i_row,:,:]-clip_SH_FLUXCOM[i_row,:,:]
                im2 = ax.contourf(plot_data.lon,plot_data.lat,plot_data,
                                 levels=bounds2,cmap=cmap2,vmin=vmn2,vmax=vmx2,extend='both')
                ax.set(title='',xlabel='')
                ax.set_ylabel('Latitude\n \n'+month[str(i_row+1)], fontsize=15)
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
                ax.yaxis.set_tick_params(labelright=True)

if i_row>6:
    fig2.savefig(wrk_dir+'Monthly SH FLUXCOM_CLM5 over India (2000 to 2014)_Jan to Jun_'+str(date.today())+'.png', 
                  dpi=600, bbox_inches="tight") 
else:        
    fig2.savefig(wrk_dir+'Monthly SH FLUXCOM_CLM5 over India (2000 to 2014)_Jul to Dec_'+str(date.today())+'.png', 
                  dpi=600, bbox_inches="tight")
#%% Plotting colorbars
figcbar,axcbar = plt.subplots(nrows=3,ncols=4)
cbar = figcbar.colorbar(im,ax=axcbar[:,3],shrink=0.8)
cbar.set_label('SH (kgC/m\u00b2/mon)')
cbar.set_ticks(np.linspace(vmn,vmx,4))

figcbar2,axcbar2 = plt.subplots(nrows=3,ncols=4)
cbar2 = figcbar2.colorbar(im2,ax=axcbar2[:,3],shrink=0.8)
cbar2.set_label('SH (kgC/m\u00b2/mon)')
cbar2.set_ticks(np.linspace(vmn2,vmx2,7))