#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 16:04:15 2024

@author: knreddy
"""

#%% Import libraries
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
from matplotlib.lines import Line2D
from datetime import date
import string
from cmcrameri import cm
#%% Loading CLM5 data
data_dir = '/Users/knreddy/Documents/GMD_Paper/Data_for_plotting_Figures/'

CLM_Wheat_LAI_matfile = sio.loadmat(data_dir+'SW_Sitescale_LAI_Data_6Feb.mat')
Wheat_Def_Site_LAI = CLM_Wheat_LAI_matfile["CLM_SiteLAI_Default"]
Wheat_Mod1_Site_LAI = CLM_Wheat_LAI_matfile["CLM_SiteLAI_Modified1"]
Wheat_Mod2_Site_LAI = CLM_Wheat_LAI_matfile["CLM_SiteLAI_Modified2"]

CLM_Rice_LAI_matfile = sio.loadmat(data_dir+'Rice_Sitescale_LAI_Data_6Feb.mat')
Rice_Def_Site_LAI = CLM_Rice_LAI_matfile["CLM_SiteLAI_Default"]
Rice_Mod1_Site_LAI = CLM_Rice_LAI_matfile["CLM_SiteLAI_Modified1"]
Rice_Mod2_Site_LAI = CLM_Rice_LAI_matfile["CLM_SiteLAI_Modified2"]

LAI_Wheat_sites2 = pd.DataFrame({'Site':['Cooch_Behar', 'Faizabad','Jobner',
                                        'Ludhiana','Meerut','Nadia','Pantnagar','Parbhani'],
              'site_ID':['COB','FAZ','JOB','LUD','MEE','NAD','PAN','PAR'],
              'year':[[2000,2001],[2002,2003,2004],[2013],[2011,2012],[2011,2012,2013],
                      [2000,2001,2002,2008,2009,2013],[2007,2008],[2001,2005,2009]]})

# Wheat_CLM_Obs_LAI_df = pd.read_excel(data_dir+'SiteScale_Data/Wheat_CLM_Obs_LAI.xlsx',index_col=0)
# Wheat_LAI_df extracted from Varma et al.(2024)
#Comparing_ISAM_CLM5_sitescale_simulations.py in /Users/knreddy/Documents/PhD_Thesis/ISAM_CLM5_Comparison
# Wheat_LAI_df['Obs_LAI'][1][2][0] = np.nan # correcting for repeating value
#%% Plotting 
fig,ax = plt.subplots(nrows=4, ncols=6,layout='constrained',sharex=True,sharey=True,figsize=(15,12))
xaxis_lim = [305,486]
row_count=0
col_count=0
for i_site_count, i_site in enumerate(LAI_Wheat_sites2.Site):
    clm_def_lai = Wheat_LAI_df['CLM_Def_LAI'][i_site_count]
    clm_mod1_lai = Wheat_LAI_df['CLM_Mod1_LAI'][i_site_count]
    clm_mod2_lai = Wheat_LAI_df['CLM_Mod2_LAI'][i_site_count]
    obs_lai = Wheat_LAI_df['Obs_LAI'][i_site_count]
    obs_doy = Wheat_LAI_df['Obs_doy'][i_site_count]
    for i_years in np.arange(len(obs_lai)):
            year_obs = obs_doy[i_years]
            year_obs[year_obs<200] = year_obs[year_obs<200]+365
            # if any((obs_lai[i_years] >10) or (obs_lai[i_years]<0)):
            #     print(i_site,i_years)
            ax[row_count,col_count].plot(np.arange(90,480),clm_def_lai[i_years][0][90:480],c=cm.batlow(10))
            ax[row_count,col_count].plot(np.arange(90,480),clm_def_lai[i_years][1][90:480],'--',c=cm.batlow(10))
            ax[row_count,col_count].plot(np.arange(90,480),clm_mod1_lai[i_years][0][90:480],c=cm.batlow(120))
            ax[row_count,col_count].plot(np.arange(90,480),clm_mod1_lai[i_years][1][90:480],'--',c=cm.batlow(120))
            ax[row_count,col_count].plot(np.arange(90,480),clm_mod2_lai[i_years][0][90:480],c=cm.batlow(200))
            ax[row_count,col_count].plot(np.arange(90,480),clm_mod2_lai[i_years][1][90:480],'--',c=cm.batlow(200))
            ax[row_count,col_count].scatter(year_obs,obs_lai[i_years],c='k')
            ax[row_count,col_count].set_title(i_site+' (SY:'+str(LAI_Wheat_sites2.year[i_site_count][i_years])+')')
            ax[row_count,0].set_ylabel('LAI (m\u00b2/m\u00b2)')
            ax[row_count,col_count].annotate('('+list(string.ascii_lowercase)[row_count*6+col_count]+')',
                                             xy=(100,5),xytext=(100,5),fontsize=14)
            if col_count<5:
                col_count=col_count+1
                row_count=row_count
            else:
                col_count = 0
                row_count = row_count+1

for i_col in np.arange(col_count,6):
    ax[row_count,i_col].set_axis_off()
    
ax[-1,0].set_xlim([91,480])
ax[-1,0].set_xticks([91,151,213,274,335,397,457]);
ax[-1,0].set_xticklabels(['Apr','Jun','Aug','Oct','Dec','Feb','Apr']);

legend_elements = [Line2D([0], [0], color=cm.batlow(10), label='CLM5_Def'),
                    Line2D([0], [0], color=cm.batlow(120),label='CLM5_Mod1'),
                    Line2D([0], [0], color=cm.batlow(200), label='CLM5_Mod2'),
                    Line2D([0], [0], marker='o', color='w', label='Obs',
                          markerfacecolor='k', markersize=10)
                    ]

legend_elements2 = [Line2D([0], [0], c='k',label='Rainfed'),
                    Line2D([0], [0], c='k',label='Irrigated',linestyle='--'),
                    ]

legend1 = ax[3,4].legend(handles=legend_elements,loc='upper right',fontsize=16)
ax[3,4].add_artist(legend1)
legend2 = ax[3,5].legend(handles=legend_elements2,loc='upper right',fontsize=16)
ax[3,5].add_artist(legend2)

ax[3,4].annotate('* SY- Sowing year',xy=(170, clm_def_lai[-1][0][140]+1),
            xytext=(170, clm_def_lai[-1][0][140]+1),fontsize=14)

savefig_dir = '/Users/knreddy/Documents/GMD_Paper/Figures/'
fig.savefig(savefig_dir+'Figure_3_Sitescale_evaluation_wheat_CLM5_'+str(date.today())+'.png',
dpi=600, bbox_inches="tight")




