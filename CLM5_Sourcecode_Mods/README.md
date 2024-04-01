# Details of the directory
This directory contains the source codes of the CLM5 modifications made for the study published with title "**Improving the representation of major Indian crops in Community Land Model version 5.0 (CLM5) using site-scale crop data**"
## CNPhenology.F90
-> The modified CNPhenology.F90 is available as CNPhenologyMod.F90. Be cautious while using it in your simulations to change the name to CNPhenology.F90 or else CLM model will show an error.
-> code added starting at line 2001 to incorporate the sowing of a crop in same year as harvested.
## CropType.F90
There are three files available here:
### Croptype_Default.F90
-> The default file available in CESM2.1.1 release
### Croptype_Mod1.F90
-> Modified to incorporate the user-specified latvary_intercept and latvary_slope
### Croptype_Latvary.F90
-> Included rice to have latitude variation in base temperature
