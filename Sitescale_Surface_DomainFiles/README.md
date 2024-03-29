# Creating domain and surface files (for IITD lat, lon)
## 1. Go to mkmapdata directory in the cesm version installed
	cd cesm2.1.1/components/clm/tools/mkmapdata
### Take interactive nodes
## 2. Creating a scripgrid file (for example IITD)
	./mknoocnmap.pl -p 28.5457,77.1928 -n IITD
### This creates two SCRIPgrid files in ../mkmapgrids/. (eg. SCRIPgrid_IITD_noocean_c240223.nc)
	export GRIDFILE=~/cesm2.1.1/components/clm/tools/mkmapgrids/SCRIPgrid_IITD_noocean_c240223.nc
	export PTNAME=IITD
## 3. Creating the map files
	./mkmapdata.sh -r $PTNAME -f $GRIDFILE -t regional -l
### This creates map files in mkmapdata
	export MAPFILE=/~/cesm2.1.1/components/clm/tools/mkmapdata/”mapfile created in earlier step”	
	export CDATE=%yy%mm%dd (date of the mapfiles created)
## 4. Creating the domain files
	./gen_domain -m $MAPFILE -o $PTNAME -l $PTNAME
## 5. Creating the Surface files
	./mksurfdata.pl -r usrspec -usr_gname $PTNAME -usr_gdate $CDATE -usr_mapdir $MAPFILE -l $CESMDATAROOT -y 2000 -crop	


