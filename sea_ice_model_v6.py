import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

#Variables et constantes---------------------------------------------------------------------------------------------------------------------
#Physical constants
Lfus = 3.35e5           #Latent heat of fusion for water [J/kg]
rhoi = 917              #Sea ice density [kg/m3]
ki = 2.2                #Sea ice thermal conductivity [W/m/K]
ks = 0.31               #Snow thermal conductivity [W/m/K]
sec_per_day = 86400     #Second in one day [s/day]
epsilon = 0.99          #surface emissivity
sigma = 5.67e-8         #Stefan-Boltzmann constantes
Kelvin = 273.15         #Conversion from Celsius to Kelvin           

#Bottom boundary condition
T_bo = -1.8          #Bottom temperature [C]



#conditions initiales
thick_0 = 0.1       #épaisseur initiale [m]
T_0 = -10           #température initiale [C]
Q_w = 5             #flux de chaleur océanique [W/m^2]
snow_0 = 0.00       #initiale snow thickness [m]
nbry = 5            #nombre d'année souhaité
alb = 0.8           #surface albedo

h_w = 50            #épaisseur de la couche mélangée [m]
C_w = 4000          #Capacité calorifique de l'eau [J/kg/K]
alb_w = 0.1         #albedo de l'eau 
rhow = 1025         #densité de l'eau [kg/m^3]

alb_s = 0.8         #albedo de la neige
rhos = 330          #densité de la neige [kg/m^3]

N_d = nbry*365      #nombre de jour dans x années



#Fonction de calcul des flux de surface-----------------------------------------------------------------------------------------------------------
#Other physical constants




Q_SOL = np.zeros(N_d)       #flux de chaleur solaire
Q_NSOL = np.zeros(N_d)      #flux de chaleur non solaire

def solar_flux(day):
    
    Q_sol= 314 * np.exp((-(day-164)**2)/4608)
    return(Q_sol)

def non_solar_flux(day):
    
    Q_nsol = 118 * np.exp((-0.5*(day-206)**2)/(53**2))+179
    return(Q_nsol)
    
    
#Fonction qui calcul la température de surface de la glace----------------------------------------------------------------------------------------
T_0K = T_0 + Kelvin        #conversion en Kelvin
T_boK = T_bo + Kelvin      #conversion en Kelvin
x0 = 200.15                #valeur de départ N-R

def f(T_su,h,ki,epsilon,sigma,alb,day):
    return (-(h*epsilon*sigma)/ki*T_su**4 - T_su + h/ki*((solar_flux(day))*(1-alb)+non_solar_flux(day))+T_boK)
 
def df(T_su,h,ki,epsilon,sigma,alb,day):
    return (-(4*h*epsilon*sigma)/ki*T_su**3-1)


def get_root(h,ki,epsilon,sigma,alb,day):
    root = optimize.newton(f,x0,fprime = df, args= (h,ki,epsilon,sigma,alb,day))
    return(root)
    
     

#Definition des vecteurs--------------------------------------------------------------------------------------------------------------------------
thick_temp = np.zeros(N_d) #Epaisseur de la glace 
thick_temp[0] = thick_0
Q_csurf = np.zeros(N_d) #Flux de chaleur en surface [W/m^2]
ROOT_surf = np.zeros(N_d) #Température de surface de la glace [K]
Q_net = np.zeros(N_d) #Flux de chaleur net [W/m^2]
doy = (np.arange(0,N_d))%365+1 #vecteur multi-year 1-365 pour chaque année
T_w = np.zeros(N_d) #Température de la couche de mélange [K] 
T_w[0] = T_boK  
h_s = np.zeros(N_d) #Epaisseur de neige
h_s[0] = 0
h_d = np.zeros(N_d) #Tirge de la glace (ice draft) 
deltaH = np.zeros(N_d) #Différence de hauteur entre le tirage de la glace+neige et l'épaisseur de glace

#Fonction qui calcul les chutes de neige dans l'année----------------------------------------------------------------------------------------------
def get_snowfall(day,h_s):
    
    if day >= 232 and day <= 303: 
        snowfall = h_s +   0.3/71
    elif day >= 305 and day <= 365:
        snowfall = h_s +  0.05/181
    elif day >= 1 and day <= 120:
        snowfall = h_s + 0.05/181
    elif day >= 121 and day <= 151:    
        snowfall = h_s +  0.05/31
    else:
        snowfall = h_s
    return(snowfall)
 
 
#Fonction qui calcul l'épaisseur de glace en tenant compte de la température de la couche de mélange et de la neige---------------------------------------------------------------
def get_thick_temp(T_boK, ki, ks, rhoi, Lfus, snow_0, Q_w, epsilon,sigma, alb, N_d):

    for j in range(0,N_d-1):
    
        doy = (np.arange(0,N_d))%365+1   #vecteur année 1-365 pour chaque année
        
        if thick_temp[j] > 0:    #-----------------------------------------partie où il y a tjrs de la glace--------------------------------------
            T_w[j] = T_boK                  
            ROOT_surf[j] = get_root(thick_temp[j],ki,epsilon,sigma,alb,doy[j])
            h_d[j] = (rhos*h_s[j] + rhoi*thick_temp[j])/rhow  
            deltaH[j] = h_d[j] - thick_temp[j]                           
            
            if ROOT_surf[j] >= 273.15:
                ROOT_surf[j] = 273.15
                h_s[j+1] = get_snowfall(doy[j],h_s[j])
                
                if h_s[j] > 0:                                                          #fonte de la neige
                    Q_net[j] = (1-alb_s)*solar_flux(doy[j]) + non_solar_flux(doy[j]) - epsilon*sigma*273.15**4                       
                    h_s[j+1] = get_snowfall(doy[j],h_s[j]) - (Q_net[j])*sec_per_day/(rhos*Lfus)
                    Q_csurf[j] = (ki*ks*(thick_temp[j]+h_s[j]))/(ki*h_s[j]+ks*thick_temp[j]) * (ROOT_surf[j]-T_boK)/(thick_temp[j]+h_s[j])
                    thick_temp[j+1] = thick_temp[j] - (Q_csurf[j]+Q_w)*sec_per_day/(rhoi*Lfus) 
                    if deltaH[j] > 0:
                        thick_temp[j+1] = thick_temp[j+1] + deltaH[j]
                        h_s[j+1] = h_s[j+1] - deltaH[j]
                elif h_s[j] <= 0:                                                       #pas de neige => fonte de la glace de mer 
                    h_s[j] = 0
                    Q_csurf[j] = ki * (ROOT_surf[j]-T_boK)/(thick_temp[j])    
                    Q_net[j] = (1-alb)*solar_flux(doy[j]) + non_solar_flux(doy[j]) - epsilon*sigma*273.15**4
                    thick_temp[j+1] = thick_temp[j] - (Q_csurf[j]+Q_w+Q_net[j])*sec_per_day/(rhoi*Lfus)

            else:
                h_s[j+1] = get_snowfall(doy[j],h_s[j])  
                if thick_temp[j] >= 0.1:                                                #formation de la glace de mer avec neige 
                    if h_s[j] < 0:
                        h_s[j] = 0
   
                    Q_csurf[j] = (ki*ks*(thick_temp[j]+h_s[j]))/(ki*h_s[j]+ks*thick_temp[j]) * (ROOT_surf[j]-T_boK)/(thick_temp[j]+h_s[j])    
                    thick_temp[j+1] =  thick_temp[j] - (Q_csurf[j]+Q_w)*sec_per_day/(rhoi*Lfus) 
                    if deltaH[j] > 0:
                        thick_temp[j+1] = thick_temp[j+1] + deltaH[j]
                        h_s[j+1] = h_s[j+1] - deltaH[j]                     
                    
                else:                                                                   #formation de la glace de mer sans neige 
                    if h_s[j] < 0:
                        h_s[j] = 0
                    Q_csurf[j] = ki * (ROOT_surf[j]-T_boK)/(thick_temp[j])
                    thick_temp[j+1] =  thick_temp[j] - (Q_csurf[j]+Q_w)*sec_per_day/(rhoi*Lfus)
           
        if thick_temp[j] <= 0:     #-------------------------------------partie où il n'y a pas tjrs de la glace-----------------------------------------------------
            thick_temp[j] = 0
            
            Q_net[j] = (1-alb_w)*solar_flux(doy[j]) + non_solar_flux(doy[j]) - epsilon*sigma*273.15**4 #flux de chaleur absorbé par l'océan 
            h_s[j] = 0                                                                  #set la neige à 0 car elle tombe dans l'eau
            
            T_w[j] =  T_w[j-1] + (Q_net[j])*sec_per_day/(rhow*C_w*h_w) 
            if T_w[j] > T_boK:                                                          #temperature non suffisante (trop haute) pour faire de la glace 
                thick_temp[j+1] = 0
            else:
                T_w[j] = T_boK
                thick_temp[j+1] = thick_temp[j] - (Q_net[j])*sec_per_day/(rhoi*Lfus)    #pour relancer la glace
                if thick_temp[j+1] >=0.1:
                    h_s[j+1] = get_snowfall(doy[j],h_s[j]) 
            
           

        
       
    
        
    return(thick_temp, ROOT_surf, T_w, h_s) 



    



#Sortie des vecteurs et figures---------------------------------------------------------------------------------------------------------------------------------------------

thick_temp = get_thick_temp(T_boK, ki, ks, rhoi, Lfus, snow_0, Q_w, epsilon,sigma, alb, N_d)[0]  
ROOT_surf = get_thick_temp(T_boK, ki, ks, rhoi, Lfus, snow_0, Q_w, epsilon,sigma, alb, N_d)[1]
T_w = get_thick_temp(T_boK, ki, ks, rhoi, Lfus, snow_0, Q_w, epsilon,sigma, alb, N_d)[2]
h_s = get_thick_temp(T_boK, ki, ks, rhoi, Lfus, snow_0, Q_w, epsilon,sigma, alb, N_d)[3]

ROOT_surf[-1] = get_root(thick_temp[-1], ki, epsilon,sigma,alb,doy[-1])                                                     #dernière case du vecteur température de surface de la glace 
Q_csurf[-1] = (ki*ks*(thick_temp[-1]+snow_0))/(ki*snow_0+ks*thick_temp[-1]) * (ROOT_surf[-1]-T_boK)/(thick_temp[-1]+snow_0) #dernière case du vecteur de flux de chaleur de surface 
T_w[-1] =  T_w[-2] + (Q_net[-1])*sec_per_day/(rhow*C_w*h_w)                                                                 #dernière case du vecteur de la température de la couche de mélange

day_x = np.arange(1,N_d+1) #vecteur temps


fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(day_x, thick_temp)
axs[0, 0].set_title("Ice thickness")
axs[0, 0].set_xlabel("Time[day]")
axs[0, 0].set_ylabel("[m]")
axs[1, 0].plot(day_x, h_s)
axs[1, 0].set_title("Snow depth")
axs[1, 0].set_xlabel("Time[day]")
axs[1, 0].set_ylabel("[m]")
axs[0, 1].plot(day_x, ROOT_surf)
axs[0, 1].set_title("Surface Temperature")
axs[0, 1].set_xlabel("Time[day]")
axs[0, 1].set_ylabel("Temperature[K]")
axs[1, 1].plot(day_x, T_w)
axs[1, 1].set_title("Water temperature")
axs[1, 1].set_xlabel("Time[day]")
axs[1, 1].set_ylabel("Temperature[K]")
fig.tight_layout()
plt.show()






