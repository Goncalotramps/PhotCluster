# PhotCluster
We have created a code to do the photometry of clusters and astronomical structures, using different observation images and filters in the same sky region and to analyse them.

There's 2 important files, one that does the observation reduction and correction, photometry and magnitude calculation, as well as the HR diagram and the isochrone fitting (phot_cluster.py) and another one that the user can use in order to obtain a estimative of the luminosity and mass of each star of that cluster, as well as possible approximations of their bolometric corrections (phot_study.py). This one also does the mass and luminosity distributions of the cluster and the luminosity and mass map according to the observed sky area.

The user should put all the .fits files in the same folder that he puts the .py file. The isochrones .txt file must be obtained in http://stev.oapd.inaf.it/cgi-bin/cmd_3.3 and the user must clear the # put in the line where the parameters names are.

If you use the code, please cite it as it is in citation.cff. It would be great. 
