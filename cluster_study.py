from curses import BUTTON3_RELEASED
from http.client import BAD_GATEWAY
from math import cos
import numpy as np
import astropy
import photutils
from astropy.table import Table
from astropy.io import fits
from astropy.wcs import wcs
import astropy.units as u
from matplotlib.pyplot import *
from scipy import interpolate
from shapely.geometry import LineString, Point
from astropy.stats import sigma_clipped_stats
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import math

import matplotlib.pyplot as plt

import os
output_dir = 'photometry'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def overwrite(filename):
    if os.path.isfile(filename):
    # If it exists, remove it (overwrite)
        os.remove(filename)

# Now, save the new image
    plt.savefig(filename, dpi=500)

from astropy.visualization import astropy_mpl_style

plt.style.use(astropy_mpl_style)

from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename 

#This part will be improved adding the results obtained with phot_cluster.py in a text file and then read it to get them but right know we need to copy them from the terminal

#insert the array of values obtained in phot_cluster for

b_mag=[] #B apparent magnitude

b_mag_err=[] #B apparent magnitude error

g_mag=[] #V apparent magnitude 

g_mag_err=[] #B apparent magnitude error

b_g=[] #color index

b_g_err=[] #color index error 

par=[] #parallax

ra=[] 

dec=[]


l_lsun=[] 
m_app=[]
m_msun=[]
d_sun=149597871e3
logmass=[]
        
i=0

cresc=np.sort(g_mag)

output_dir = 'clusterstudy'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

######################################################################
############## GETTING THE BOLOMETRIC CORRECTION #####################
######################################################################

#getting some real data of bolometric correction in order of B_V index
bc1=[-4,-2.8,-1.5,-0.4,-0.12,-0.06,0,-0.03]
bg1=[-0.35,-0.31,-0.16,0,0.13,0.27,0.42,0.58]
bc2=[-0.06,0,-0.03,-0.07,-0.2,-0.6,-1.2,-2.3]
bg2=[0.27,0.42,0.58,0.70,0.89,1.18,1.45,1.63]

bc1=np.asarray(bc1)
bg1=np.asarray(bg1)
bc2=np.asarray(bc2)
bg2=np.asarray(bg2)


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def parabola(x,a,b,c,d):  #those are the functions that we found to fit the best the data points (its a inovating approach that we are testing and that seems to be good)
    #y=x*x*a+b*x+c
    y=-(a/(np.exp(-d*x)+b))+x+c
    return y

def exp(x,a,b,c,d,e):
    y=-(a/(np.exp(d*x)+b))-e*x+c
    return y

parameters1, covariance1 = curve_fit(exp, bg1, bc1)
parameters2, covariance2 = curve_fit(parabola, bg2, bc2)

a1 = parameters1[0]
b1= parameters1[1]
c1=parameters1[2]
d1= parameters1[3]
e1=parameters1[4]
a2 = parameters2[0]
b2= parameters2[1]
c2= parameters2[2]
d2= parameters2[3]

fit_par1=exp(bg1,a1,b1,c1,d1,e1) #here we adjust the function to the points
fit_par2 = parabola(bg2, a2, b2,c2,d2)

plt.scatter(bg1, bc1, color= "blue", #here we do the plot just to see if it seems good
            marker= "o", s=25)
plt.scatter(bg2, bc2, color= "blue", 
            marker= "o", s=25)

plt.plot(bg1, fit_par1, '-',label="B_C vs B$_{corr}$ Proposal Function for B_C<0.58",color="red")

plt.plot(bg2, fit_par2, '-',label="B_C vs B$_{corr}$ Proposal Function for for B_C>0.58",color="violet")

plt.ylim(-4.2, 2)

plt.legend()

plt.xlabel('B_V')
# frequency label

plt.ylabel("B$_{correction}$") 


filename=os.path.join(output_dir, f'bc_vs_bv.png')

overwrite(filename)

plt.clf() 

while i<len(g_mag):

    if b_g[i]<0.58: #here we get the bolometric corrections for B_V in each interval (the functions are different)

        bc=-(a1/(np.exp(d1*b_g[i])+b1))-e1*b_g[i]+c1

    if b_g[i]>0.58:

        bc=-(a2/(np.exp(-d2*b_g[i])+b2))+b_g[i]+c2

    if i==5: #We are using a real value for the WR133 
        bc=-3.0264

    m_abs_g=g_mag[i]+5*(math.log10(par[i]*0.001)+1)+bc #here we get the bolometric magnitude in visual filter for each star 

    l_lsun.append(pow(10,-0.4*(m_abs_g - 71.197425))/(3.846e26)) #here we get the luminosity in lsun

    m_msun.append(pow(l_lsun[i]*(1/1.4), (2/7))) #here we get the luminosity and here we get the aproximate mass in msun

    logmass.append(math.log10(m_msun[i]))

    i=i+1

print("L/Lsun")
print(l_lsun)
print("M/Msun")
print(m_msun)
print("Log10(M/Msun)")
print(logmass)


output_dir = 'clusterstudy'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

from matplotlib import pyplot as plt
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

###################################################################
############ MASS DISTRIBUTION AND IMF ############################
###################################################################
 
iqr = np.percentile(logmass, 75) - np.percentile(logmass, 25) # we use the Freedman-Diaconis rule to see the best number of bins to distribute our mass data 
numbins = int((max(logmass) - min(logmass)) / (2 * iqr / (len(logmass) ** (1/3))))

print("Num Bins obtained by Freedman-Diaconis rule ")        
print(numbins)

print("Max logmass")
print(np.max(logmass))
print("Min logmass")
print(np.min(logmass))

divisoes=[numbins]

max=2 #here we select the upper and lower limits for the mass values to do the distribution
min=0.2

best_erro=1000 #using the best_erro and the divisoes list we could do some diferent mass dist with different number of bins and see which one was closer to IMF slope
full_int=max-min


for div in divisoes: #we are going to create an histogram (using the )
    dN = np.zeros(div)

    int=full_int/div

    def imf(x,alpha,c): #here we define a linear function to compare the alpha with the IMF slopes
        y =-alpha*x+c
        return y
    i=0


    for mass in logmass:
        part=min
        i=0
        while part<max:
            if part <= mass < part + int:
                dN[i]=dN[i]+1
            part = part + int
            i=i+1
    i=0
    bins=[]

    part=min
    while part<max:
        bins.append(((int/2)+part))
        part=part+int
        
    num_mass=[]

    while i<len(dN):
        if dN[i]/int>0:
            num_mass.append(math.log10(dN[i]/int))
        elif dN[i]/int==0:
            num_mass.append(0)
        i=i+1

    err=[]
    i=0
    while i<len(dN):
        err.append(np.std(num_mass))
        i=i+1
    
    bins=np.asarray(bins)

    num_mass=np.asarray(num_mass)
    
    parameters, covariance = curve_fit(imf, bins, num_mass)

    fit_A = parameters[0]
    fit_B = parameters[1]

    perr=np.sqrt(np.diag(covariance[0])[0])

    print("PARAMETERS FIT IMF") #we get the fit parameters
    print(fit_A)
    print(perr)

    plt.bar(bins,num_mass,width=int,align='center',color="lightskyblue") # A bar chart

    fit_y = imf(bins, fit_A, fit_B)
    fit_salpeter=imf(bins,2.35,fit_B)
    fit_kroupa=imf(bins,2.3,fit_B)

    plt.scatter(bins, num_mass, color= "blue",
                marker= "o", s=25)


    plt.plot(bins, fit_y, '-',label="IMF simple fit $\\alpha =0.703 \\pm 0.177$ ",color="red") #this example is with the parameters that we obtained

    err1=[err,err]

    plt.errorbar(bins, num_mass, yerr=err1, capsize=2,markersize=5,fmt='o', color='blue',ecolor='grey',zorder=2)

    plt.legend()

    erro=abs(fit_A-2.3)


    if best_erro > erro: #in case of trying diferent divisions
        best_erro=erro
        best_div=div
    
    plt.xlim(0,2.2)
    plt.ylim(0, 2.2)

    plt.xlabel('log10(M (M_sun))')

    plt.ylabel('dN/dM') 

    plt.show()

    filename=os.path.join(output_dir, f'estimate_masses.png')

    overwrite(filename)

    plt.clf() 

print("Lowest error to the IMF slope and best division: ")
print(best_erro)
print(best_div)

##########################################################################
############# TURN OFF POINT MASS ESTIMATIVE #############################
##########################################################################

i=0.8 #we set a initial B_V value where we see that the turn off of the main sequence has already started to take the mean value of the masses of that region untill a upper limit
b_g_toff=[]
while i<1.2: #here we set the upper limit where its clear that it isnt in the turn off point
    b_g_toff.append(i)
    i=i+0.05 #we get masses from 0.05 to 0.05 in B_V values

mean_d=np.mean(par)

l_lsun_toff=[]
m_msun_toff=[]
logmass_toff=[]
err_m_msun_toff=[]

i=0

#Here we calculate the error of the mass value obtained for the turn off point

while i<len(b_g_toff):
    bc=-(a2/(np.exp(-d2*b_g[i])+b2))+b_g[i]+c2
    term1=0.00001*np.sqrt(abs(np.diag(covariance2)[0]))/(np.exp(-d2*b_g[i])+b2)
    term2=a2*b_g[i]*np.exp(-d2*b_g[i])*0.000001*np.sqrt(abs(np.diag(covariance2)[3]))/((np.exp(-d2*b_g[i])+b2)*(np.exp(-d2*b_g[i])+b2))
    term3=(1-(a2*d2*np.exp(-d2*b_g[i]))/((np.exp(-d2*b_g[i])+b2)*(np.exp(-d2*b_g[i])+b2)))*b_g_err[i]
    term4=a2*np.sqrt(abs(np.diag(covariance2)[1]))*0.00001/((np.exp(-d2*b_g[i])+b2)*(np.exp(-d2*b_g[i])+b2))

    errbc=np.sqrt(term1*term1+term2*term2+term3*term3+term4*term4+0.001*0.001*np.sqrt(abs(np.diag(covariance2)[2]))*np.sqrt(abs(np.diag(covariance2)[2])))
    m_abs_g=b_g_toff[i]+5*(math.log10(mean_d*0.001)+1)+bc

    err_m_abs_g=np.sqrt((5*0.0000000000000001/(np.log(10)*mean_d))*(5*0.0000000000000001/(np.log(10)*mean_d))+errbc*errbc)


    l_lsun_toff.append(pow(10,-0.4*(m_abs_g - 71.197425))/(3.846e26))
    
    err_l_lsun_toff= (err_m_abs_g*0.4*np.log(10)*pow(10,-0.4*(m_abs_g - 71.197425)))/(3.846e26)

    m_msun_toff.append(pow(l_lsun_toff[i]*(1/1.4), (2/7)))
    

    err_m_msun_toff.append(pow(1/1.4,2/7)*(2/7)*pow(l_lsun_toff[i],-5/7)* err_l_lsun_toff)
    logmass_toff.append(math.log10(m_msun_toff[i]))
    i=i+1


m_msun_toff_mean=np.mean(m_msun_toff)
i=0

sum=0

for err in  err_m_msun_toff:
    sum=sum+err*err

m_msun_toff_mean_err=np.sqrt(sum)

print("Turn off point mass and error estimatives:")
print(m_msun_toff_mean)
print(m_msun_toff_mean_err)

#######################################################
######### LUMINOSITY COLOR MAP ########################
#######################################################

cm1 = matplotlib.cm.get_cmap('autumn')
#cm1 = cm1.reversed()
cm2 = matplotlib.cm.get_cmap('summer') 
cm2 = cm2.reversed()
cm3 = matplotlib.cm.get_cmap('winter')
cm3 = cm3.reversed() 
cm4 = matplotlib.cm.get_cmap('hot')
cm4 = cm4.reversed() 

ra1=[]
dec1=[]
ra2=[]
dec2=[]
ra3=[]
dec3=[]
ra4=[]
dec4=[]
l1=[]
l2=[]
l3=[]
l4=[]
while i<len(l_lsun): #we divide the stars in luminosity intervals and save the ra and dec of those stars
    if l_lsun[i]<=100:
        ra1.append(ra[i])
        dec1.append(dec[i])
        l1.append(l_lsun[i])
    elif 1000>l_lsun[i]>100:
        ra2.append(ra[i])
        dec2.append(dec[i])
        l2.append(l_lsun[i])
    elif 1000<l_lsun[i]<10000:
        ra3.append(ra[i])
        dec3.append(dec[i])
        l3.append(l_lsun[i])
    elif 10000<l_lsun[i]:
        ra4.append(ra[i])
        dec4.append(dec[i])
        l4.append(l_lsun[i])
    if i==5:
        ra4.append(ra[i])
        dec4.append(dec[i])
        l4.append(l_lsun[i])
        
    i=i+1

fig = plt.figure(figsize=(20,10))

sc = plt.scatter(ra1,dec1, c=l1, vmin=0, vmax=100, s=35, cmap=cm1,zorder=2) #here we plot them to see their distribution on the color map
sc2 = plt.scatter(ra2,dec2, c=l2, vmin=100, vmax=1000, s=35,cmap=cm2,zorder=2)
sc3 = plt.scatter(ra3,dec3, c=l3, vmin=1000, vmax=10000, s=35,cmap=cm3,zorder=2)
plt.scatter(ra4,dec4, c="indigo",label="Stars with L(Lsun)>10000 ($L_\\odot$)", s=35,cmap=cm4,zorder=2)

plt.colorbar(sc)
plt.colorbar(sc2)
plt.colorbar(sc3)

plt.xlabel('RA (Degrees)')
# frequency label

plt.ylabel('DEC (Degrees)') 

plt.title('position_vs_l_lsun')
# showing legend

overwrite(filename)

plt.clf()  

##################################################
########LUMINOSITY DISTRIBUTION ##################
##################################################

 #We do nearly the same that we did for mass distribution but here we dont adjust any function

best_erro=1000

div=7
log10lum=[]
for lum in l_lsun:
    log10lum.append(math.log10(lum))

max=np.max(log10lum)#max(logmass)

min=np.min(log10lum)#min(logmass)

full_int=max-min

dN = np.zeros(div)

int=full_int/div

for lum in log10lum:
    part=min
    i=0
    while part<max:
        if part <= lum < part + int:
            dN[i]=dN[i]+1
        part = part + int
        i=i+1
i=0
bins=[]

part=min
while part<max:
    bins.append(((int/2)+part))
    part=part+int
    
err=[]
i=0

while i<len(dN):
    err.append(np.std(dN))
    i=i+1

bins=np.asarray(bins)

N=np.asarray(dN)

plt.bar(bins,N,width=int,align='center',color="lightskyblue") # A bar chart


plt.scatter(bins, N, color= "blue",
            marker= "o", s=25)


err1=[err,err]

plt.errorbar(bins, N, yerr=err1, capsize=2,markersize=5,fmt='o', color='blue',ecolor='grey',zorder=2)


plt.xlim(0,np.max(log10lum)+0.5)
plt.ylim(-2, 17.5)

plt.xlabel('log10(L/L$_\\odot$)')
# frequency label

plt.ylabel('N Stars') 

plt.show()

filename=os.path.join(output_dir, f'estimate_lum.png')

overwrite(filename)

plt.clf() 

###################################################################
######### MASS INTERVALS IN SKY AREA ##############################
##################################################################

ra1=[]
dec1=[]
ra2=[]
dec2=[]
ra3=[]
dec3=[]
ra4=[]
dec4=[]
l1=[]
l2=[]
l3=[]
l4=[]

i=0

while i<len(m_msun):
    if m_msun[i]<=5:
        ra1.append(ra[i])
        dec1.append(dec[i])
        l1.append(m_msun[i])
    elif 10>m_msun[i]>5:
        ra2.append(ra[i])
        dec2.append(dec[i])
        l2.append(m_msun[i])
    elif 10<m_msun[i]<25:
        ra3.append(ra[i])
        dec3.append(dec[i])
        l3.append(m_msun[i])
    elif 40<m_msun[i]<60:
        ra4.append(ra[i])
        dec4.append(dec[i])
        l4.append(m_msun[i])
        
    i=i+1

fig = plt.figure(figsize=(20,10))


plt.scatter(ra1,dec1,color="red" ,label="Star with M <5 $M_\\odot$",s=58,zorder=2)
plt.scatter(ra2,dec2,color="green" ,label="Star with 5 $M_\\odot$ < M < 10 $M_\\odot$", s=58,zorder=2)
plt.scatter(ra3,dec3,color="violet"  ,label="Star with 10 $M_\\odot$ < M < 25 $M_\\odot$", s=58,zorder=2)
plt.scatter(ra4,dec4,color="blue"  ,label="Star with 40 $M_\\odot$ < M < 60 $M_\\odot$", s=58,zorder=2)

plt.title('position_vs_M(Msun)')
# showing legend
plt.legend(fontsize="20",loc ="center right")

plt.xlabel('RA (Degrees)')
# frequency label

plt.ylabel('DEC (Degrees)') 

filename=os.path.join(output_dir, f'position_vs_M(Msun).png')

overwrite(filename)

plt.clf()  




