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

b_mag=[0.920578397822178, 11.813873670662353, 6.972693229077212, 9.953527482375524, 13.912946532605563, 13.090040039259865, 7.6617971193580665, 11.268226403329715, 13.284827797627813, 14.015652136163546, 11.861349095687858, 13.283026260328617, 13.539461404319523, 13.061952839645512, 14.31068850582824, 12.047946688446423, 9.784575186961627, 8.054230559980606, 12.688419759042414, 12.294079660313237, 12.294080754011064, 12.491385282924679, 11.452515390432744, 11.537554608461836, 11.495316170635599, 8.430257415366105, 11.620339581289137, 7.5562545948696975, 11.20047106197556, 6.335719751813668, 9.302674691545729, 12.581981138732942, 12.568539154995507, 7.01796631616779, 12.664081397003066, 10.733948805361335, 12.444776080142072]

b_mag_err=[1.3605720291410448, 1.360852068850293, 1.3605002462951512, 1.3605121909293434, 1.3779186449009364, 1.3640770909755973, 1.3605003620149974, 1.3606256424450913, 1.3658059215364264, 1.3871593290580204, 1.3609522084707222, 1.366145844160692, 1.3696692216029684, 1.3655240944988947, 1.410868781650656, 1.361174177700301, 0.7074239083499635, 0.7073459982499055, 0.7215685377515471, 0.7142987966946881, 0.7142988205225178, 0.7572528907590589, 0.7089323849101176, 0.7090836370077412, 0.709100639706344, 0.7073493093809687, 0.7095047157551976, 0.707344291731473, 0.7083571603393937, 0.7073430920467297, 0.7073756568195572, 0.7234732098319051, 0.7200295242673747, 0.7073434592822703, 0.7220225890219701, 0.7077235077332259, 0.715639172296842]

g_mag=[10.576659395579371, 11.511512121161276, 6.849486898065093, 9.655836261351798, 13.331238437517769, 12.834407659249017, 7.4304293144032805, 10.99934953891179, 12.732178339583282, 13.615911784384565, 11.586029260294366, 12.889538720885207, 13.174529670834598, 12.727789033292325, 13.949298668854935, 11.593096500798762, 10.025288392857496, 8.130351744245642, 12.75972040978401, 12.266429728166463, 12.250053484406024, 12.539863821908607, 11.729083403208469, 11.80564118486234, 11.76766279761653, 8.892274927681232, 11.896771665770963, 7.916597047009134, 11.275412630307398, 6.581199450653584, 9.765324403977559, 12.901791397151303, 12.568995954272612, 7.430420039812329, 12.886692167743476, 10.738904203105182, 12.727726359010266]

g_mag_err=[0.8162426716945315, 0.8163546891395089, 0.8162167666172057, 0.8162695144842719, 0.8199204128231576, 0.8350979829695164, 0.8162168146628505, 0.816267064055679, 0.8175213806798479, 0.8243892531836317, 0.8163488962537314, 0.818852349062315, 0.819433270193376, 0.8184655374183205, 0.8359472158087875, 0.8164252809518032, 0.42461129111735857, 0.4243342795324582, 0.47259970742033447, 0.4734918379936031, 0.4393931022686571, 0.4492802789033569, 0.8165445922509423, 0.8164983186592035, 0.8174573410740285, 0.8162296462226203, 0.8165204899391089, 0.8162182737014011, 0.42708059038615453, 0.42432580506531403, 0.8162219369765176, 0.8180122515761552, 0.45337404204188725, 0.8162168146642679, 0.8180043908272607, 0.42524606973337825, 0.8181870351812582]


b_g=[0.34391900224280647, 0.30236154950107696, 0.12320633101211875, 0.2976912210237259, 0.5817080950877944, 0.255632380010848, 0.23136780495478604, 0.26887686441792447, 0.5526494580445309, 0.3997403517789806, 0.2753198353934927, 0.3934875394434094, 0.3649317334849247, 0.3341638063531871, 0.36138983697330573, 0.4548501876476614, -0.24071320589586875, -0.07612118426503578, -0.07130065074159475, 0.027649932146774248, 0.04402726960504033, -0.04847853898392884, -0.2765680127757246, -0.2680865764005045, -0.2723466269809318, -0.46201751231512667, -0.27643208448182577, -0.3603424521394363, -0.07494156833183929, -0.24547969883991527, -0.46264971243182984, -0.31981025841836086, -0.0004567992771047358, -0.41245372364453914, -0.2226107707404097, -0.004955397743847456, -0.2829502788681939]

b_g_err=[1.586634282238981, 1.5869320501439836, 1.5865594001726544, 1.5865967798857248, 1.6034117610051792, 1.5994045614803662, 1.5865595241217108, 1.5866928054104221, 1.5917810851912557, 1.6136383253249047, 1.5870149445282553, 1.5927566157710351, 1.5960778993853315, 1.5920244623115833, 1.6399263601317193, 1.5872445877732018, 0.8250717148524047, 0.8248623776282232, 0.8625610924025564, 0.8569815001539064, 0.8386233381686528, 0.8805025324065965, 1.0813557220029886, 1.0814199501791302, 1.0821553593212792, 1.0800804973956513, 1.081712832580471, 1.0800686169726414, 0.8271443025788643, 0.8248555259623077, 1.0800919267648463, 1.0920428238275142, 0.8508763352062982, 1.0800669691876021, 1.0910764420839938, 0.8256553658893684, 1.0870002067453377]

par=[0.5197564986163528, 0.46812666412739956, 0.5380353041993117, 0.4722753516207746, 0.4951110217652847, 0.5175660096227219, 0.5498643860966984, 0.506825541958511, 0.4994096181593436, 0.5150272957480986, 0.5175180469202163, 0.5284300390972912, 0.5223526362843269, 0.49096905139628905, 0.539137622411168, 0.49916686154607004, 0.5379939384027261, 0.5262706079950656, 0.5043761761936325, 0.5149319884149137, 0.5149319884149137, 0.5201720645179146, 0.4849454486961847, 0.46821908895964504, 0.47792602601405143, 0.545879740746754, 0.4782159091228425, 0.5204174132706191, 0.46812666412739956, 0.5380353041993117, 0.4722753516207746, 0.5175660096227219, 0.5175660096227219, 0.5498643860966984, 0.5284300390972912, 0.5241988368066384, 0.49096905139628905]

ra=[301.41106001331826, 301.49801629143053, 301.488834489061, 301.4854910289137, 301.38680084931906, 301.49785371692076, 301.49461259852416, 301.4575408953329, 301.3949206085345, 301.51785546664416, 301.450975166726, 301.4769833265481, 301.4568902746811, 301.5334731919839, 301.4841676937907, 301.37141301676974, 301.46936147019323, 301.64567524286923, 301.501211401271, 301.6016628042328, 301.6016628042328, 301.51366292945437, 301.51578486687316, 301.4862810089736, 301.5171903492186, 301.49988803992755, 301.4894776017431, 301.50565301168916, 301.49801629143053, 301.488834489061, 301.4854910289137, 301.49785371692076, 301.49785371692076, 301.49461259852416, 301.4769833265481, 301.5610468602117, 301.5334731919839]

dec=[35.75870793110902, 35.78609481941231, 35.78834615644555, 35.78992059788903, 35.77435477227547, 35.796159037984914, 35.797161333655275, 35.793876529658334, 35.78516250570025, 35.80691087301645, 35.801002045882015, 35.83163765226666, 35.83826654855599, 35.85487348689841, 35.856089654813125, 35.84195811051553, 35.70678727641109, 35.74060116468971, 35.7223217734846, 35.74508666820821, 35.74508666820821, 35.73624297132084, 35.760999338491345, 35.755918159884125, 35.76487447970893, 35.76231068049866, 35.760771036779566, 35.76550052639097, 35.78609481941231, 35.78834615644555, 35.78992059788903, 35.796159037984914, 35.796159037984914, 35.797161333655275, 35.83163765226666, 35.85076824804854, 35.85487348689841]


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
        bc=-4.5

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

divisoes=[8]

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




