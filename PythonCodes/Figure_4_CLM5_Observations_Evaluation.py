#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 10:42:12 2024

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
#%% Function for calculating bias
def bias_data(model,obs):
    x = model
    y=obs
    bias = np.abs((np.nansum(x) - np.nansum(y)))/np.nansum(y)
    return bias
#%% creating a geodataframe data for all variables
def cal_bias(i_crop,i_var,df_data):
    model_Def_var = 'CLM5_Def_'+i_var
    model_Mod1_var = 'CLM5_Mod1_'+i_var
    model_Mod2_var = 'CLM5_Mod2_'+i_var
    obs_var = 'Obs_'+i_var
    CLM_var_Def_bias = np.empty(np.shape(df_data.Site))
    CLM_var_Mod1_bias = np.empty(np.shape(df_data.Site))
    CLM_var_Mod2_bias = np.empty(np.shape(df_data.Site))
    for i_num, i_site in enumerate (df_data.Site):
        CLM_var_Def_bias[i_num] = bias_data(eval(df_data[model_Def_var][i_num]),eval(df_data[obs_var][i_num]))
        CLM_var_Mod1_bias[i_num] = bias_data(eval(df_data[model_Mod1_var][i_num]),eval(df_data[obs_var][i_num]))
        CLM_var_Mod2_bias[i_num] = bias_data(eval(df_data[model_Mod2_var][i_num]),eval(df_data[obs_var][i_num]))
    
    model_var_bias_data = np.stack((CLM_var_Def_bias,CLM_var_Mod1_bias,CLM_var_Mod2_bias))
    return model_var_bias_data
#%% ranking the bias in model simulations
def rank_bias(data_df):    
    r_bias_array = np.empty([3,len(data_df)])
    a = data_df['CLM5_Def_bias']
    b = data_df['CLM5_Mod1_bias']
    c = data_df['CLM5_Mod2_bias']
    test_array = np.array([a,b,c])
    for i_row in np.arange(len(a)):
        r_bias_array[:,i_row] = rankdata(test_array[:,i_row]*-1)
        rank_df = pd.DataFrame({'CLM5_Def_rank':r_bias_array[0,:],
                            'CLM5_Mod1_rank':r_bias_array[1,:],
                            'CLM5_Mod2_rank':r_bias_array[2,:],})
    new_df = pd.concat([data_df,rank_df],axis=1,join='inner')
    
    return new_df
#%% loading rice site information
LAI_Rice_sites = pd.DataFrame({'Site':['Ananthapur', 'Hyderabad','Jabalpur','Kaul','Kuthulia','Pantnagar','Raipur'],
              'site_ID':['RED','HYD','JAB','KAU','KUT','PAN','RAI'],
              'year':[[2000,2001],[2010],[2009,2010,2011],[2008],[2013],[2011,2012],[2009]]})

GY_Rice_sites = pd.DataFrame({'Site':['Ananthapur','Faizabad','Hyderabad',
                                      'Jabalpur','Kaul','Kuthulia','Pantnagar','Raipur'],
              'site_ID':['RED','FAZ','HYD','JAB','KAU','KUT','PAN','RAI'],
              'year':[[2000,2001],[2000,2001],[2010],[2009,2010,2011],
                      [2008],[2013],[2010,2011,2012],[2009,2012]]})

GSL_Rice_sites = pd.DataFrame({'Site':['Ananthapur','Faizabad','Hyderabad',
                                      'Jabalpur','Kaul','Kuthulia','Pantnagar','Raipur'],
              'site_ID':['RED','FAZ','HYD','JAB','KAU','KUT','PAN','RAI'],
              'year':[[2000,2001],[2000,2001],[2010],[2009,2010,2011],
                      [2008],[2013],[2010,2011,2012],[2009,2012]]})

data_dir = '/Users/knreddy/Documents/GMD_Paper/Data_for_plotting_Figures/'
#%% Loading data and creating wheat geodataframe
Rice_GSL_df = pd.read_excel(data_dir+'SiteScale_Data/Rice_GSL.xlsx',index_col=0)
Rice_LAI_df = pd.read_excel(data_dir+'SiteScale_Data/Rice_LAI.xlsx',index_col=0)
Rice_GY_df = pd.read_excel(data_dir+'SiteScale_Data/Rice_GY.xlsx',index_col=0)

Wheat_LAI_df = pd.read_excel(data_dir+'SiteScale_Data/Wheat_LAI.xlsx',index_col=0)
Wheat_GY_df = pd.read_excel(data_dir+'SiteScale_Data/Wheat_GY.xlsx',index_col=0)
Wheat_GSL_df = pd.read_excel(data_dir+'SiteScale_Data/Wheat_GSL.xlsx',index_col=0)

Rice_GSL_bias_data = cal_bias('Rice','GSL', Rice_GSL_df)
Rice_LAI_bias_data = cal_bias('Rice','LAI', Rice_LAI_df)
Rice_GY_bias_data = cal_bias('Rice','GY', Rice_GY_df)

Wheat_LAI_bias_data = cal_bias('Wheat','LAI', Wheat_LAI_df)
Wheat_GY_bias_data = cal_bias('Wheat','GY', Wheat_GY_df)
Wheat_GSL_bias_data = cal_bias('Wheat','GSL', Wheat_GSL_df)
#%% Normalise all bias
Rice_bias_data = np.stack((np.insert(Rice_LAI_bias_data,1,np.nan,axis=1),Rice_GY_bias_data,Rice_GSL_bias_data))
Wheat_bias_data = np.stack((Wheat_LAI_bias_data,Wheat_GY_bias_data,Wheat_GSL_bias_data))

all_bias_data = np.stack((Rice_bias_data,Wheat_bias_data))

all_bias_data_norm = (all_bias_data - np.nanmin(all_bias_data)) / (np.nanmax(all_bias_data) - 
                                                                                np.nanmin(all_bias_data))

Rice_LAI_bias_data_norm = np.delete(all_bias_data_norm[0,0,:],1,axis=1)
Rice_GY_bias_data_norm = all_bias_data_norm[0,1,:]
Rice_GSL_bias_data_norm = all_bias_data_norm[0,2,:]

Wheat_LAI_bias_data_norm = all_bias_data_norm[1,0,:]
Wheat_GY_bias_data_norm = all_bias_data_norm[1,1,:]
Wheat_GSL_bias_data_norm = all_bias_data_norm[1,2,:]
#%% creating GFD data for plotting
crop = ['Rice','Wheat']
variable = ['LAI','GY','GSL']

for i_c_num,i_crop in enumerate(crop):
    for i_v_num,i_var in enumerate(variable):
        df_name = i_crop+'_'+i_var+'_Bias_df'
        bias_data_name = i_crop+'_'+i_var+'_bias_data_norm'
        locals()[df_name] = pd.DataFrame({'CLM5_Def_bias':eval(bias_data_name)[0,:],
                                          'CLM5_Mod1_bias':eval(bias_data_name)[1,:],
                                          'CLM5_Mod2_bias':eval(bias_data_name)[2,:],
                                          })
        or_df_name = i_crop+'_'+i_var+'_df'
        var_new_inter = i_var+'_'+i_crop+'_sites_new'
        var_new_df = i_crop+'_'+i_var+'_gfd'
        locals()[var_new_inter] = pd.concat([eval(or_df_name)['Site'],eval(or_df_name)['Lat'],
                                             eval(or_df_name)['Lon'],eval(df_name)],
                                         axis=1,join='inner')
        locals()[var_new_df] = gpd.GeoDataFrame(
            eval(var_new_inter),geometry=gpd.points_from_xy(eval(var_new_inter).Lon,
                                                            eval(var_new_inter).Lat),crs="EPSG:4326")
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
#%% Plotting overlapping markers
fig,ax= plt.subplots(ncols = 2, nrows = 3, figsize=(9,13),sharex=True,sharey=True,layout='constrained')
fsize=14

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
var=['LAI','GY','GSL']
crop_m = ['SW','Ri']

for i_c_n,i_crop in enumerate(crop):
    for i_v_n,i_var in enumerate(var):
        axes = ax[i_v_n,i_c_n]
        India.plot(ax = axes,facecolor='none',edgecolor='black')
        crop_map = 'clipped_CFT_'+crop_m[i_c_n]+'_2000_14'
        plot_data = eval(crop_map)
        im = axes.contourf(plot_data.lon,plot_data.lat,plot_data,levels=bounds,cmap=cmap)        
        data = i_crop+'_'+i_var+'_gfd'
        rank_data = rank_bias(eval(data))
        scat_name1 = 'scat1_'+str(i_c_n)+'_'+str(i_v_n)
        scat_name2 = 'scat2_'+str(i_c_n)+'_'+str(i_v_n)
        scat_name3 = 'scat3_'+str(i_c_n)+'_'+str(i_v_n)

        for i_num,i_site in enumerate(eval(data).Site):
            locals()[scat_name1] = axes.scatter(eval(data)["Lon"][i_num],eval(data)["Lat"][i_num],marker = 'o',
                               s=eval(data)['CLM5_Def_bias'][i_num]*300,zorder=rank_data['CLM5_Def_rank'][i_num],
                               c='red',alpha=1,label='CLM5_Def',edgecolor='none')
            locals()[scat_name2] = axes.scatter(eval(data)["Lon"][i_num],eval(data)["Lat"][i_num],marker = 'o',
                               s=eval(data)['CLM5_Mod1_bias'][i_num]*300,zorder=rank_data['CLM5_Mod1_rank'][i_num],
                               c='cyan',alpha=1,label='CLM5_Mod1',edgecolor='none')
            locals()[scat_name3] = axes.scatter(eval(data)["Lon"][i_num],eval(data)["Lat"][i_num],marker = 'o',
                               s=eval(data)['CLM5_Mod2_bias'][i_num]*300,zorder=rank_data['CLM5_Mod2_rank'][i_num],
                               c='blue',alpha=0.8,label='CLM5_Mod2',edgecolor='none')
                
        if i_crop == 'Wheat':            
            for i_num, i_site in enumerate(eval(data)['Site']):
                if i_site == 'Jobner':
                    axes.annotate(i_site,xy=(eval(data)['Lon'][i_num],eval(data)['Lat'][i_num]),
                                xytext=(eval(data)['Lon'][i_num]-8,eval(data)['Lat'][i_num]+2),
                                arrowprops=arrowprops_kw, bbox=bbox_kw)
                elif i_site == 'Faizabad' :
                    axes.annotate(i_site,xy=(eval(data)['Lon'][i_num],eval(data)['Lat'][i_num]),
                                xytext=(eval(data)['Lon'][i_num]-12,eval(data)['Lat'][i_num]-2), 
                                arrowprops=arrowprops_kw, bbox=bbox_kw)
                elif (i_site == 'Meerut') :
                    axes.annotate(i_site,xy=(eval(data)['Lon'][i_num],eval(data)['Lat'][i_num]),
                                xytext=(eval(data)['Lon'][i_num]+1,eval(data)['Lat'][i_num]+5), 
                                arrowprops=arrowprops_kw, bbox=bbox_kw)
                elif (i_site == 'Ludhiana') :
                    axes.annotate(i_site,xy=(eval(data)['Lon'][i_num],eval(data)['Lat'][i_num]),
                                xytext=(eval(data)['Lon'][i_num]-8,eval(data)['Lat'][i_num]+2), 
                                arrowprops=arrowprops_kw,bbox=bbox_kw)
                elif (i_site == 'Cooch_Behar'):
                    axes.annotate(i_site,xy=(eval(data)['Lon'][i_num],eval(data)['Lat'][i_num]),
                                xytext=(eval(data)['Lon'][i_num]-1,eval(data)['Lat'][i_num]+3), 
                                arrowprops=arrowprops_kw, bbox=bbox_kw)
                elif (i_site == 'Pantnagar'):
                    axes.annotate(i_site,xy=(eval(data)['Lon'][i_num],eval(data)['Lat'][i_num]),
                                    xytext=(eval(data)['Lon'][i_num]+2,eval(data)['Lat'][i_num]+3), 
                                    arrowprops=arrowprops_kw, bbox=bbox_kw)
                else:
                    axes.annotate(i_site,xy=(eval(data)['Lon'][i_num],eval(data)['Lat'][i_num]),
                                xytext=(eval(data)['Lon'][i_num]+4,eval(data)['Lat'][i_num]-2), 
                                arrowprops=arrowprops_kw, bbox=bbox_kw)
        if i_crop == 'Rice':
            for i_num, i_site in enumerate(eval(data)['Site']):
                if i_site == 'Kuthulia':
                    axes.annotate(i_site,xy=(eval(data)['Lon'][i_num],eval(data)['Lat'][i_num]),
                                xytext=(eval(data)['Lon'][i_num]-10,eval(data)['Lat'][i_num]+2),
                                arrowprops=arrowprops_kw, bbox=bbox_kw)
                elif ((i_site == 'Anathapur') or (i_site == 'Hyderabad')):
                    axes.annotate(i_site,xy=(eval(data)['Lon'][i_num],eval(data)['Lat'][i_num]),
                                xytext=(eval(data)['Lon'][i_num]+3,eval(data)['Lat'][i_num]), 
                                arrowprops=arrowprops_kw, bbox=bbox_kw)
                elif (i_site == 'Kaul'):
                    axes.annotate(i_site,xy=(eval(data)['Lon'][i_num],eval(data)['Lat'][i_num]),
                                xytext=(eval(data)['Lon'][i_num]-1.2,eval(data)['Lat'][i_num]+4), 
                                arrowprops=arrowprops_kw, bbox=bbox_kw)
                elif (i_site == 'Raipur'):
                    axes.annotate(i_site,xy=(eval(data)['Lon'][i_num],eval(data)['Lat'][i_num]),
                                xytext=(eval(data)['Lon'][i_num]-10,eval(data)['Lat'][i_num]-2), 
                                arrowprops=arrowprops_kw, bbox=bbox_kw)
                else:
                    axes.annotate(i_site,xy=(eval(data)['Lon'][i_num],eval(data)['Lat'][i_num]),
                                xytext=(eval(data)['Lon'][i_num]+4,eval(data)['Lat'][i_num]), 
                                arrowprops=arrowprops_kw, bbox=bbox_kw)
        ########## Panel. No#####################
        panel_no = list(string.ascii_lowercase)[i_v_n]+'.'+str(i_c_n+1)
        axes.annotate(panel_no,xy=(67.25,36),fontsize=16)
                
legend_elements = [Line2D([0], [0], marker='o', color='w', label='CLM5_Def',
                          markerfacecolor='red', alpha=1, markersize=10),
                   Line2D([0], [0], marker='o', color='w', label='CLM5_Mod1',
                          markerfacecolor='cyan', alpha=1, markersize=10),
                   Line2D([0], [0], marker='o', color='w', label='CLM5_Mod2',
                          markerfacecolor='blue', alpha=0.8, markersize=10)
                   ]
legend1 = ax[2,1].legend(handles=legend_elements,loc='lower right',fontsize="large")
ax[2,1].add_artist(legend1)

scat1 = ax[1,1].scatter(Rice_GY_gfd["Lon"],
                        Rice_GY_gfd["Lat"],
                        marker = 'o',s=Rice_GY_gfd['CLM5_Def_bias']*300,
                        c='grey',edgecolor='none',zorder=-1)
kw = dict(prop="sizes", num=6, color='black', fmt="{x:0.2f}",
          func=lambda s: s/300)
legend2 = ax[1,1].legend(*scat1.legend_elements(**kw),
                    loc="lower right", title="MAB",
                    fontsize="large")
legend2._legend_box.align= 'center'
ax[1,1].add_artist(legend2)

ax[0,1].annotate('max. LAI',xy=(87.25,35),fontsize=16,bbox=dict(boxstyle="square",fc="white",lw=0.1))
ax[0,0].annotate('max. LAI',xy=(87.25,35),fontsize=16,bbox=dict(boxstyle="square",fc="white",lw=0.1))
ax[1,0].annotate('Yield',xy=(87.25,35),fontsize=16,bbox=dict(boxstyle="square",fc="white",lw=0.1))
ax[1,1].annotate('Yield',xy=(87.25,35),fontsize=16,bbox=dict(boxstyle="square",fc="white",lw=0.1))
ax[2,0].annotate('GS Length',xy=(87.25,35),fontsize=16,bbox=dict(boxstyle="square",fc="white",lw=0.1))
ax[2,1].annotate('GS Length',xy=(87.25,35),fontsize=16,bbox=dict(boxstyle="square",fc="white",lw=0.1))

ax[2,0].set_xticks(np.arange(70,105,10))
# ax[1,0].set_xticklabels(np.arange(70,105,10),fontsize=fsize)
ax[2,1].set_xticks(np.arange(70,105,10))
# ax[1,1].set_xticklabels(np.arange(70,105,10),fontsize=fsize)
ax[0,0].set_yticks(np.arange(5,40,10))
# ax[0,0].set_yticklabels(np.arange(5,40,10),fontsize=fsize)
ax[1,0].set_yticks(np.arange(5,40,10))
ax[2,0].set_yticks(np.arange(5,40,10))
# ax[1,0].set_yticklabels(np.arange(5,40,10),fontsize=fsize)
ax[2,0].set_xlabel('Longitude [\u00b0E]',fontsize=fsize)
ax[2,1].set_xlabel('Longitude [\u00b0E]',fontsize=fsize)
ax[0,0].set_ylabel('Latitude [\u00b0N]',fontsize=fsize)
ax[1,0].set_ylabel('Latitude [\u00b0N]',fontsize=fsize)
ax[2,0].set_ylabel('Latitude [\u00b0N]',fontsize=fsize)
ax[0,0].set_title('1. Wheat',fontsize=fsize+4)
ax[0,1].set_title('2. Rice',fontsize=fsize+4)

cbar = fig.colorbar(im,ax=ax[:,1],shrink=0.5,fraction=0.04,pad=0.001)
cbar.set_label('Crop Area (% of grid cell)',fontsize=fsize+2)
cbar.set_ticks(np.linspace(0,100,6))
cbar.set_ticklabels(np.linspace(0,100,6),fontsize=fsize-2)

savefig_dir = '/Users/knreddy/Documents/GMD_Paper/Figures/'
fig.savefig(savefig_dir+'Figure_4_Sitescale_evaluation_wheat_rice_'+str(date.today())+'.png', 
              dpi=600, bbox_inches="tight")