CDF       
      n         ni        nj        nv              title         CESM domain data:      Conventions       CF-1.0     source_code       CSVN $Id: gen_domain.F90 65202 2014-11-06 21:07:45Z mlevy@ucar.edu $    SVN_url       _ $URL: https://svn-ccsm-models.cgd.ucar.edu/tools/mapping/gen_domain/trunk/src/gen_domain.F90 $    Compiler_Optimized        TRUE   hostname      login03    history       )created by cez218275, 2023-09-10 23:03:58      source        �/scratch/civil/phd/cez218275/my_cesm_sandbox/components/clm/tools/mkmapdata/map_0.25x0.25_MODIS_to_Meerut_nomask_aave_da_c230910.nc    map_domain_a      m/scratch/civil/phd/cez218275/cesm2_2_0_mksurf/lnd/clm2/mappingdata/grids/SCRIPgrid_0.25x0.25_MODIS_c170321.nc      map_domain_b      o/scratch/civil/phd/cez218275/my_cesm_sandbox/components/clm/tools/mkmapgrids/SCRIPgrid_Meerut_nomask_c230910.nc    map_grid_file_ocn         m/scratch/civil/phd/cez218275/cesm2_2_0_mksurf/lnd/clm2/mappingdata/grids/SCRIPgrid_0.25x0.25_MODIS_c170321.nc      map_grid_file_atm         o/scratch/civil/phd/cez218275/my_cesm_sandbox/components/clm/tools/mkmapgrids/SCRIPgrid_Meerut_nomask_c230910.nc          xc                    	long_name         longitude of grid cell center      units         degrees_east   bounds        xv          	�   yc                    	long_name         latitude of grid cell center   units         degrees_north      bounds        yv          	�   xv                       	long_name          longitude of grid cell verticies   units         degrees_east         	�   yv                       	long_name         latitude of grid cell verticies    units         degrees_north            	�   mask                  	long_name         domain mask    note      unitless   coordinates       xc yc      comment       $0 value indicates cell is not active        	�   area                  	long_name         $area of grid cell in radians squared   coordinates       xc yc      units         radian2         	�   frac                  	long_name         $fraction of grid cell that is active   coordinates       xc yc      note      unitless   filter1       =error if frac> 1.0+eps or frac < 0.0-eps; eps = 0.1000000E-11      filter2       Jlimit frac to [fminval,fmaxval]; fminval= 0.1000000E-02 fmaxval=  1.000000          	�@SZ�G�{@=ffffff@SW�z�H@S^z�G�@S^z�G�@SW�z�H@=Y�����@=Y�����@=s33332@=s33332   >�C#��Z?�      