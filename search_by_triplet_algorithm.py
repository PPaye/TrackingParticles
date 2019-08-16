#!/usr/bin/env python
# coding: utf-8

# In[129]:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from split import *
from score import *
from scipy import interpolate
import time 
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings 
warnings.simplefilter('ignore', np.RankWarning)
import matplotlib.pyplot as plt
import matplotlib

"""
Title:       Search_By_Triplet + TIME_RESOLUTION + two new cuts. (THE SEARCH OF TRIPLETS WILL BE DIFFERENT
Status:      Stable 

Autor:       Piter Amador Paye Mamani.

Description:
             This python-code is an implementation of the algorithm Search By Triplet that was inspired on the work of
             Daniel Campora plus a litle modifications.   

Technical Details:
             Per time the algorithm spend a time around 100 seconds on average. 
Changes :
        * 
        * 
        * 
        * Now it is a pythonfile. It is not a notebook
        * Put dr in function of theta_scatter
        * Use dr instead of dx and dy [doing]                                 
        * Reviewing the units                                           
        * Making changes on the time.                                   
        * Finally, the time information is added.                   
        * A very good code without time is completly developed.    
        * New restrictions are applied at the seeding algorithm.   
        * 
        * Implementing the z cut. [doing]
        * Change np.arctan() by np.arctan2()

1.  Squeletum of the algorithm
2.  First Tracks. 
5.  Adding exceptions. 
6.  I've deleted unnecessary comments. Also, I was getting an error at time to compute findcandidatewindows.
    Problems. One get the values of tracks 
7.  Changing the jerarquy of the function, according to the paper. It means that findcandidatewindows is calculated
    on all modules befero they were processed.
6.  
7.  
8.  I've added the information of weak_tracks and I've added the information of USED and NOt USED 
9.  dphi  is a constant value
9.  Adding timing information 

10. In this version I will plot a graphic of efficiency in function of dphi. 
    Here, I am not concentrating on the plots of the tracks. Only on the plots of the efficiency that depend on dphi.
    In other words. I have to run the main program and get the values of the efficiency and then plot. 
    I am thinking on work only with 0.004 percent of the data. Because it is more fast than all data. 
    
11. Adding TIMING TO THE ALGORITHM.     
    To add timing information to the algorithm 
    I've do it previoues analysis like see if the time difference follow a gaussian distribution.
    Plus another important question is to see. How we can add the new restriction to our values. 
    
    I refer that I have to compute the difference in time between t1 and t2 to see if the values surpasses a new technological approach. 
12. The timing information was added to this algorithm. However, the serach by triplet algorithm is falling according to (probably) 
    the lost of one variable. It means that when we are using phi, it depend on the x and y. However, the plots that I am getting are good 
    only for a xy superposed planes. In other words, the window variable on phi is a good candidate but it lost informatin on Z X and Z Y.
13. Most possible changes.
"""


# # Defining FUNCTIONS

# In[130]:


def rho(x,y):
    return np.sqrt(x*x + y*y)
def r(x,y,z):
    return np.sqrt(x*x + y*y + z*z)
def theta(x,y,z):
    return np.arccos(z/r(x,y,z))
def phi(x,y):
    return np.arctan2(y/x)
def module(r):
    return np.sqrt(np.sum(r*r))
def r_e(z, r_l, r_c):
    z_c = r_c[2] 
    r_versor = (r_l - r_c)/module(r_l - r_c)               # computing r_versor
    r_versor_dot_z_versor = r_versor[2]  
    return r_c - r_versor/r_versor_dot_z_versor*(z_c - z)  # HAVE ACCOUNT THE MINUES SUGN.
def correct_time(hit_time, x, y, z):
    c = 0.299792 # Light Velocity in [mm/ps]  
    travel_time = np.sqrt(x*x + y*y + z*z)/c
    return hit_time - travel_time

GRALDR1 = []
GRALDR2 = []
def DR(delta, dx, dy, dz): 
    if delta == "inf":
        return 10.
    global theta_scatt
    dr1 = 3*np.sqrt(dx*dx + dy*dy)
    GRALDR1.append(dr1)
    dr2 = delta*np.tan(theta_scatt)  
    GRALDR2.append(dr2)    
    return max(dr1, dr2)

# In[131]:

def reading_data(name, fraction, event):
    global time_resolution           # Adding a time resolution to our analysis of tracks
    
    """
    EVENT
    55microns50psInner55microns50psOuter_EventNumber.txt
    
    25microns0psInner200microns50psOuter_test.txt
    25microns0psInner200microns50psOuter_train.txt
    25microns75psInner25microns75psOuter_test.txt
    25microns75psInner25microns75psOuter_train.txt
    55microns0psInner55microns0psOuter_test.txt
    55microns0psInner55microns0psOuter_train.txt
    55microns100psInner200microns50psOuter_test.txt
    55microns100psInner200microns50psOuter_train.txt
    55microns50psInner55microns50psOuter_test.txt
    55microns50psInner55microns50psOuter_train.txt
    
    RAMPData25microns0psInner200microns50psOuter_test.txt
    RAMPData25microns0psInner200microns50psOuter_train.txt
    RAMPData25microns75psInner25microns75psOuter_test.txt
    RAMPData25microns75psInner25microns75psOuter_train.txt
    RAMPData55microns0psInner55microns0psOuter_test.txt
    RAMPData55microns0psInner55microns0psOuter_train.txt
    RAMPData55microns100psInner200microns50psOuter_test.txt
    RAMPData55microns100psInner200microns50psOuter_train.txt
    RAMPData55microns50psInner200microns50psOuter_test.txt
    RAMPData55microns50psInner200microns50psOuter_train.txt
    RAMPData55microns50psInner55microns50psOuter_test.txt
    RAMPData55microns50psInner55microns50psOuter_train.txt
    RAMPsmeared55microns200psInner55microns50psOuter_test.txt
    RAMPsmeared55microns200psInner55microns50psOuter_train.txt
    test.txt
    testTrain.txt
    """
    
    #name = 'data2/55microns50psInner55microns50psOuter_EventNumber.txt' # To be modified for others files. 
    
    df = pd.DataFrame()
    df = pd.read_csv(name, sep=' ')              # All data.
    
    columns = df.columns.values
    columns[9] = 'Event'
    df.columns = columns

    df_tmp = df.query(f'Event == {event}' ) #.copy(deep = True)  # inplace=True)
    
    df_tmp2, _ = split_frac(df_tmp, fraction)
    
    return df_tmp2
# In[132]:

def sortbyphi():
    '''Description:
    Sort each D_i increasingly accoording to phi
    And add a column to the dataframe_module with the name of used to accept or neglect hits. 
    '''
    global time_resolution  
    global df 
    global sigma_z 
    
    df['phi']   = np.arctan2(df['x'], df['y'])                        
    df['t_c']   = correct_time(df['t'], df['x'], df['y'], df['z'])        
    df['used']  = 0   
    df['delta'] = 0 # this is the distance from one module to another module -> in this sense
    
    modules = []
    z_modules = [-277.0, -252.0, -227.0, -202.0, -132.0, -62.0, -37.0, -12.0, 13.0, 38.0, 63.0, 88.0, 113.0, 138.0, 163.0, 188.0, 213.0, 238.0, 263.0, 325.0, 402.0, 497.0, 616.0, 661.0, 706.0, 751.0]
    
    for i in range(len(z_modules)-1,-1 ,-1):
        z_m = z_modules[i]
        #print(sigma_z)
        mod = df.query(f" {z_m} - {sigma_z} <= z <= {z_m} + {sigma_z}").copy(deep=True)
        mod['z_mod'] = z_m 
        
        if i == 0 : 
            mod['delta'] = abs(277.0  -252.0)  # this value is a string. It is necessary to verify it after. 
        else :
            #print(i)
            mod['delta'] =  z_modules[i] - z_modules[i-1]
            #print("the distance difference between modules is :", i, i-1, "delta: ", z_modules[i] - z_modules[i-1]) 
        #mod.loc[mod.index.values, "z_mod"] = z_m
        #mod.loc[mod.index.values, "used"]  = False
        # IMPORTANT  
        mod = mod.sort_values('phi', ascending=True) 
        #print("Index ", mod['t_c'])
        modules.append(mod)  
        
    modules.reverse()  #sort=True) # necessary step
    
    tmp_df =pd.DataFrame()
    for mod in modules:
        tmp_df = pd.concat([tmp_df, mod])
    df = tmp_df
    #print(df['t_c'])
    return modules

# In[133]:

#PHI = []
def findcandidatewindows(left_mod, mod, right_mod):
    #global left_mod, mod, right_mod
    #(left_mod, mod, right_mod ):
    global time_resolution               # Adding a time resolution to our analysis of tracks
    # phi_window     =  phi_extrapolation_base + np.abs( hit_Zs[h_center]) * phi_extrapolation_coef 
    global phi_extrapolation_coef, phi_extrapolation_base , dphi
    '''Description: 
        Compute the first and last candidates(the window) according to acceptance range(dphi) for each hit.
        SUPPOSSING THAT ALL DATA ARE ORDERED ACCOURDING TO PHI. THIS PROCCESS WAS DONE Previously
        In case of add more information to the modules, one easily can add throught the iteration 
    '''
    # CONVENTION :     
    # l_m  m  r_m   the values are ordered.      
    #  |   |   |             
    #  |   |   |    phi up  
    #  |   |   |    phi      
    #  |   |   |    phi down 
    #  |   |   |    
    right_hit_max = [] 
    right_hit_min = [] 
    
    temporal = mod['phi'] 
    
    # ITERATION OVER PHI FOR RIGHT 
    
    for phi_i in mod['phi']: 
        #print("=")
        #print(phi_i)
        if str(phi_i) == 'nan' :     
            #print(phi_i, "the value of phi_i is NaN ON RIGHT")
            m = "nan"               # minumum hit 
            M = "nan"               # maximum hit
            right_hit_min.append(m) 
            right_hit_max.append(M) 
            continue # 
        if str(phi_i) == 'NaN' :     
            #print(phi_i, "the value of phi_i is NaN ON RIGHT")
            m = "nan"               # minumum hit 
            M = "nan"               # maximum hit
            left_hit_min.append(m) 
            left_hit_max.append(M) 
            continue # 
            
        z_center = mod['z_mod'].unique()[0]
        #z = df.query(f"phi=={phi_i}")["z"].values[0]
        # GET HIT 
        """ dphi =  phi_extrapolation_base + np.abs( z_center ) * phi_extrapolation_coef """

        #PHI.append(dphi)
        down      = phi_i - dphi 
        up        = phi_i + dphi 
        #print(down, up)
        
        condition = f'{down} <= phi <=  {up}'
        tmp_df = right_mod.query(condition)
        if not tmp_df.empty:
            m = tmp_df['hit_id'][tmp_df.index[0]]     # minumum hit 
            M = tmp_df['hit_id'][tmp_df.index[-1]]    # maximum hit 
            right_hit_min.append(m) 
            right_hit_max.append(M) 
        elif tmp_df.empty :

            m = "nan" #pd.np.nan                      # minumum hit 
            M = "nan" #pd.np.nan                      # maximum hit
            right_hit_min.append(m)  
            right_hit_max.append(M) 
          
    left_hit_max = [] 
    left_hit_min = [] 
    # ITERATION OVER PHI FOR LEFT
    for phi_i in mod['phi']:
        if str(phi_i) == 'NaN' :     
            # print(phi_i, "the value of phi_i is NaN ON LEFT")
            m = "nan"               # minumum hit 
            M = "nan"               # maximum hit
            left_hit_min.append(m) 
            left_hit_max.append(M) 
            continue # 
        if str(phi_i) == 'nan' :     
            # print(phi_i, "the value of phi_i is NaN ON left")
            m = "nan"               # minumum hit 
            M = "nan"               # maximum hit
            left_hit_min.append(m) 
            left_hit_max.append(M) 
            continue # 
        # GET HIT 
        down      = phi_i - dphi 
        up        = phi_i + dphi 
        condition = f'{down} <= phi <= {up}'
        tmp_df = left_mod.query(condition)
        #print("len LEFT", len(tmp_df))
        if not tmp_df.empty :
            m = tmp_df['hit_id'][tmp_df.index[0]]        # minumum hit 
            M = tmp_df['hit_id'][tmp_df.index[-1]]       # maximum hit  
            left_hit_min.append(m)
            left_hit_max.append(M)
        elif tmp_df.empty :
            # print("data_frame is empty LEFT")
            m = "nan"               # minumum hit 
            M = "nan"               # maximum hit
            left_hit_min.append(m) 
            left_hit_max.append(M) 
            
    mod["right_hit_max"] = right_hit_max  
    mod["right_hit_min"] = right_hit_min  
    mod["left_hit_max"]  = left_hit_max   
    mod["left_hit_min"]  = left_hit_min                                                                                    
    return mod

###############################################



# In[134]:


def extrapolation_on_center_module(r_right, r_left, z_center):
    # IMPORTANT
    # the values then have the next form:
    # np(r_right), np(r_left), float(r_left)
    #modules |  |  |
    #        l  c  r
    r_versor = (r_right - r_left)/module(r_right - r_left)   #  
    distance = abs( z_center - r_left[2] )                   # 
    r_center = r_left + distance / ( np.dot(r_versor, np.array([0,0,1]) ) ) * r_versor
    return r_center 


# In[135]:


def extrapolation_to_origin(r1, r2, y):   # only works for y
                                          # in case to extrapolate for x you only need to change the values of r2 ->y2 and the unitary versor
    r_versor = (r1 - r2)/module(r1 - r2)  
    y2       = r2[1] 
    r_origin = r2 + (y2 - y)/( np.dot(r_versor, np.array([0,-1,0]) ) ) * r_versor 
    return r_origin 


# In[136]:


T_L = []
T_R = []
T_C = []

X = []
Y = []
Z = []

DR0HISTOGRAM = []
def trackseeding():
    global dr
    global sigma_z_origin
    global dx, dy
    global M_i
    global time_resolution                             # Adding a time resolution to our analysis of tracks

    global left_mod, mod, right_mod, M_i, dphi, sigma_t
    
    '''
    Description: 
        Checks the preceding and following modules for compatible hits using the above results.
        
        All triplets in the search window are fitted and compared.
        
        and the best one is kept as a track seed.
        
        stores its best found triplet
        Finding triplets is ap- plied in first instance to the modules
        that are further apart from the collision point
        Each triplet is the seed of a forming track
    '''
    
    #Necessary functions.
    def fit(triplet): 
        phi_data = [ df.query(f'hit_id == {hit}')['phi']     for hit in triplet ]
        z_data   = [ df.query(f'hit_id == {hit}')['z_mod']   for hit in triplet ]
        phi_data = [ hit.values[0] for hit in phi_data                      ]                        
        z_data   = [ hit.values[0] for hit in z_data                        ]                    
        # Kind of fit: Linear
        fitting = np.polyfit(phi_data, z_data, 1)
        chiSquared = np.sum((np.polyval(fitting, z_data) - phi_data)**2)
        return chiSquared

    df_triplets = []
    #print("error ??????", mod )
    # print("error_mod.columns:", mod.columns)
    for index, part in mod.iterrows():

        r_hit_max, r_hit_min = part["right_hit_max"], part["right_hit_min"]  
        l_hit_max, l_hit_min = part["left_hit_max"],  part["left_hit_min" ] 
        
        if  str(r_hit_max)  == "nan":
            continue 
        elif str(r_hit_min) == "nan":
            continue 
        elif str(l_hit_max) == "nan":
            continue 
        elif str(l_hit_min) == "nan":
            continue  
        if  str(r_hit_max)  == "NaN" :
            continue 
        elif str(r_hit_min) == "NaN" :
            continue 
        elif str(l_hit_max) == "NaN" :
            continue 
        elif str(l_hit_min) == "NaN" :
            continue  
        r_phi_max = right_mod.query(f"hit_id == {r_hit_max}")['phi'].values[0]   
        r_phi_min = right_mod.query(f"hit_id == {r_hit_min}")['phi'].values[0]   
                                                                                 
        l_phi_max = left_mod.query(f"hit_id == {l_hit_max}")['phi'].values[0]   
        l_phi_min = left_mod.query(f"hit_id == {l_hit_min}")['phi'].values[0]     
        
        tmp_right = right_mod.query(f"   {r_phi_min} <= phi <= {r_phi_max} & used < {flagged}  ")    # ADDING TIME
        for hit_right in tmp_right['hit_id'].values:
            tmp_left = left_mod.query(f" {l_phi_min} <= phi <= {l_phi_max} & used < {flagged}  ")    # ADDING TIME
            for hit_left in tmp_left['hit_id'].values:         
                
                #hit_left   = int( tmp_left.query( f" phi == {L}")['hit_id'].values[0]  )  
                hit_center = int( part["hit_id"] )
                #hit_right  = int( tmp_right.query(f" phi == {R}")['hit_id'].values[0]  )
                
                try :
                    r_right  = tmp_right.query(f'hit_id == {hit_right} ')[['x', 'y','z']].to_numpy()[0]
                    z_center = part['z_mod']                            
                    r_left   = tmp_left.query(f'hit_id == {hit_left} ')[['x', 'y','z']].to_numpy()[0]
                except :
                    print("here there is a problem")
                    return 1
                
                try :
                    r_center_extrapolation = extrapolation_on_center_module(r_right, r_left, z_center)
                    # MAKING A PROOF ON THE EXTRAPOLATION OF DATA. PLOTING THE VALUES OF X Y Z 
                    # make a plot of r_left and r_right and r_extrapolated
                    #r_left, r_center_extrapolation, r_right 
                    
                    x_hits = [r_left[0], r_center_extrapolation[0], r_right[0] ]
                    y_hits = [r_left[1], r_center_extrapolation[1], r_right[1] ]
                    z_hits = [r_left[2], r_center_extrapolation[2], r_right[2] ]
                    
                    verification = r_left[2] < r_center_extrapolation[2] < r_right[2]
                    if verification == False :
                        print("verifing if the value of z_center is on the ")
                        print(verification)
                    #print(temporal, type(temporal))
                except :    
                    print("here is the error ont the extrapolation==")
                    return 1 
                
                ############################################################################################################ 
                ############################################################################################################ 
                ########################################   TIMING   ######################################################## 
                ############################################################################################################ 
                ############################################################################################################ 

                # NOTATION: 't_c' is the corrected time. Against of t_c that is the time variable of the modules t_center        
                
                if time_resolution == True :
                    # print("time_resolution == True")
                    t_l =   left_mod.query(f'hit_id == {hit_left}')['t_c'].values[0]
                    t_c =      mod.query(f'hit_id == {hit_center}')['t_c'].values[0]
                    t_r = right_mod.query(f'hit_id == {hit_right}')['t_c'].values[0]

                    # CONDITIONS:
                    T_L.append(abs(t_l - t_c))
                    T_C.append(abs(t_c - t_r))
                    T_R.append(abs(t_l - t_r))
                    
                    if abs(t_l - t_c) > 3*sigma_t :
                        continue
                    if abs(t_c - t_r) > 3*sigma_t :
                        continue
                    if abs(t_l - t_r) > 3*sigma_t :
                        continue
                
                ############################################################################################################ 
                ############################################################################################################ 
                ########################################   CUT on Z   ###################################################### 
                ############################################################################################################   
                ############################################################################################################ 
                try:
                    x0 , y0,  z0 = extrapolation_to_origin(r_right, r_left, 0) 
                    #if y0 != 0 :
                    #    print("the value of y0 is :", y0)
                    #X.append(x0) 
                    #Y.append(y0) 
                    #Z.append(z0) 
                    if  abs(z0) > sigma_z_origin:
                        #print("z0 > sigma_origin: ")
                        continue 
                except : 
                    print("cut on z is the error")

                ############################################################################################################ 
                ############################################################################################################ 
                ########################################   NEW WINDOW on X and Y   ######################################### 
                ############################################################################################################ 
                ############################################################################################################
                left_cut_x  = r_center_extrapolation[0] - dx 
                right_cut_x = r_center_extrapolation[0] + dx  
                
                down_cut_y  = r_center_extrapolation[1] - dy 
                up_cut_y    = r_center_extrapolation[1] + dy       
                
                # print("left_cut_x", "right_cut_x", "down_cut_y", "up_cut_y")
                # print(left_cut_x, right_cut_x, down_cut_y, up_cut_y)
                
                try : 
                    ############################################################################################################ 
                    ############################################################################################################ 
                    ########################################   DEEP CONDITION on X and Y   ##################################### 
                    ############################################################################################################ 
                    ############################################################################################################
                    x = part['x'] 
                    y = part['y'] 
                    #print("verifying the kind of x and y ", x, y, type(x), type(y))
                    
                    new_window = mod.query(f" {left_cut_x}  < x < {right_cut_x} & {down_cut_y} < y < {up_cut_y} ").copy(deep=True) 
                    #if  (left_cut_x  < x < right_cut_x) and (down_cut_y < y < up_cut_y) :
                    x_e = r_center_extrapolation[0] 
                    y_e = r_center_extrapolation[1] 
                    
                    drValue = np.sqrt( (x - x_e )**2 + ( y - y_e )**2 )
                    # the value to change is dr. 
                    
                    #WORKING_NOW
                    #dr = DR()
                    
                    # sounds very romantic 
                    # sounds very romantic 
                    #print("here, I want to print the value of hit module")
                    #print(part)

                    try :
                        dx0    = part['dx']   
                        dy0    = part['dy']    
                        dz0    = part['dz']    
                        delta0 = part['delta'] 
                    except : 
                        print("the error is in assign variables")
                        return True
                    
                    
                    try :
                        dr0 = DR(delta0, dx0, dy0, dz0)
                    except :
                        print("DR error")
                        return True
                    DR0HISTOGRAM.append(dr0)
                    if (drValue <  dr0) : 
                        """
                        #if len(new_window) > 0 :
                        print("PROOF", part[['x', 'y', 'z']].to_numpy())
                        point = part[['x', 'y', 'z']].to_numpy()
                        plt.plot(z_hits, x_hits) 
                        plt.scatter(z_hits + [ point[2], point[2], point[2]], x_hits+[point[0], left_cut_x, right_cut_x] )
                        plt.xlabel("z")
                        plt.ylabel("x")
                        plt.show() 
                        plt.plot(z_hits, y_hits ) 
                        plt.scatter(z_hits + [ point[2], point[2], point[2]], y_hits+[point[1], down_cut_y, up_cut_y] )
                        plt.xlabel("z")
                        plt.ylabel("y")
                        plt.show()
                        """
                        pass

                    else:
                        continue 
                except : 
                    print("the new window has a syntax error ** ")
                    return
                
                #print("print the new window", len(new_window)
                

                ############################################################################################################
                ############################################################################################################
                ############################################################################################################
                ############################################################################################################
                ############################################################################################################
                
                # With this data we have built the triplets. 
                triplets = [hit_left, hit_center, hit_right] 
                
                # This a lost of memory. I mean that call by hits and not by values is a lost of memory.
                chi2 = fit(triplets)                                                                                                                                                                
                # Finally we append the values of the data to a df_triplets
                df_triplets.append(list(triplets)+[chi2])
                        
    df_triplets = pd.DataFrame(df_triplets, columns = ['left_hit', 'hit', 'right_hit', 'chi2'])  
    # Up to this point it is necessary to have the values of df_triplets complete
    # Then the algorithm should continue to get the best choices according to the values
    # of chi2. 
    
    def best_choice(df_triplets):
        seeds = []
        for hit_c in df_triplets['hit'].unique() : # UNIQUE
            # GROUPING 
            tmp = df_triplets.query(f'hit == {hit_c}')
            minimum = (tmp['chi2']).idxmin()
            t = (tmp.loc[minimum]).values     
            t = [int(i) for i in t[:3]]
            #these are the triplets       
            
            seeds.append(list(t[:3]))     # Here I am negleting the information chi2 because is not important
        return seeds                      # obviously it is a track
    
    seeds = best_choice(df_triplets)
    
    for seed in seeds:
            # #########     MARKING TRIPLES######  
            # MATCHING EACH HIT AS USED ON THE WORKING MODULE  
            hit_id_left, hit_id_center, hit_id_right = seed 
            #LEFT
            left_mod.loc[   left_mod.hit_id == hit_id_left,    "used" ]     += 1 #True
            #CENTER
            mod.loc[           mod.hit_id   == hit_id_center,  "used" ]     += 1 #True
            #RIGTH
            right_mod.loc[ right_mod.hit_id == hit_id_right,   "used" ]     += 1 #True
    return seeds


# In[ ]:

# In[137]:


DR1HISTOGRAM = []

def track_forwarding():
    global dx, dy
    global frozen_tracks                 
    global time_resolution              
    global tracks, work_module, left_mod, mod, right_mod, M_i, weak_tracks 
    global phi_extrapolation_coef, phi_extrapolation_base 
    
    new_tracks = []      
    #frozen_tracks = []   
    # Notation:
    # x0, y0, z0 is the EXTRAPOLATED track.               
    # X,  Y,  Z  is the last track on previous module.   
    # x,  y,  z  is the tracks on a window.                                                                 
    # Searching tracks on phi_e - dphi < phi < phi_e + dphi that minimize the extrapolated function.
    # r0 = np.array([x0, y0, z0] )
    # r  = np.array([x, y, 1]    )
    # R  = np.array([X,  Y,  Z ] )
    
    def module(r):
        return np.sqrt(np.sum(r*r))
    def ext_func(r0, r1, r):
        # r0, r1, r are arrays
        dx2_plus_dy2 = module(  r0-r )     # distance between hits on the working module.  
        """dz2       = module( r1-r0 )     # distance between the last two modules.                                
        return dx2_plus_dy2/dz2 
        """  
        return dx2_plus_dy2 
    
    try: 
        z_e = work_module['z_mod'].unique()[0]  # z_position of work_module  # an array  
    except :
        
        print("possible error on work_module. Probably it not have values" )
        return "error"
    
    #######################################################
    #######################################################
    ###########   PRINCIPAL LOOP OVER tracks ##############
    #######################################################
    #######################################################
    for track in tracks:          
        # print("error", time_resolution)
        #PROOF: Do you have the track values information of USED ?
        data = []   
        vector_data = []
        #EXTRAPOLATION ONLY WITH TWO LAST HITS 
        for hit in track[0:2] :
            data.append(tuple((df.query(f'hit_id == {hit}')[['phi', 'z_mod']]).values[0]))     
            vector_data.append(tuple((df.query(f'hit_id == {hit}')[['x', 'y', 'z_mod']]).values[0]))
        phi_data, z_data = zip(*data) 
        
        #EXTRAPOLATED SEGMENT FUNCTION      
        ext_seg = interpolate.interp1d(z_data, phi_data, fill_value = "extrapolate" )
        phi_e   = ext_seg( z_e )                   # an array 
        r_l, r_c = vector_data                     # THE VALUES ON LEFT AND RIGHT                                                
        r_l, r_c = np.array(r_l), np.array(r_c)    # 
        x_e, y_e, z_e = r_e(z_e, r_l, r_c)         # COMPUTING THE VALUES ON THE WORKING MODULE.  
        
        down = phi_e - dphi
        up   = phi_e + dphi
        
        #######################################################################
        ######################          TIMING        #########SWITCH########## 
        #######################################################################
        if time_resolution == True : 
            h_l = track[0]   
            h_c = track[1]   
            h_r = track[2]  
            t1 = df.query( f"hit_id == {h_l}  "   )['t_c'].values[0]   #  track      
            t2 = df.query( f"hit_id == {h_c}  "   )['t_c'].values[0]   #  track       
            t3 = df.query( f"hit_id == {h_r}  "   )['t_c'].values[0]   #  track

        if   time_resolution == True  :        # on all cases where be necessary
            df_work_module_window = work_module.query(f"{down} <= phi <= {up}  & abs(t_c - 1/3.*( {t1} + {t2} + {t3} ) ) <= 3*{sigma_t}")
        elif time_resolution == False :
            df_work_module_window = work_module.query(f"{down} <= phi <= {up}") 
    
        #Open a Window centered on phi_e: 
        z_center = mod['z_mod'].unique()[0]
        #z = df.query(f"phi=={phi_i}")["z"].values[0]
        # GET HIT 
        """ dphi =  phi_extrapolation_base + np.abs( z_center ) * phi_extrapolation_coef """
        down = phi_e - dphi
        up   = phi_e + dphi   
        #####################are down or up  NaN values?####################################
        if str(down) == 'nan' or str(down) == 'NaN' or str(up) == 'nan' or str(up) == 'NaN':
            print("An error ocurred with the values of down or up. Plese cheack.")
            break
        ############################################################################################################
        ####################################### EXTRAPOLATION TO THE ORIGIN ########################################
        ############################################################################################################ 
        ########################################   CUT on Z   ###################################################### 
        ############################################################################################################
        hit_1, hit_2, hit_3 = track[0:3]
        # work_module, left_mod, mod 
        try :
            r_right  = df.query(f'hit_id == {hit_1} ')[['x', 'y','z']].to_numpy()[0]
            z_center = df.query(f'hit_id == {hit_2} ')[['x', 'y','z']].to_numpy()[0]                            
            r_left   = df.query(f'hit_id == {hit_3} ')[['x', 'y','z']].to_numpy()[0]
        except :
            print("here there is a problem")
            return "error"
        try:
            x0 , y0,  z0 = extrapolation_to_origin(r_right, r_left, 0) 
            if  abs(z0) > sigma_z_origin:
                continue 
        except : 
            print("cut on z is the error")
        
        hit_left = track[0]   
        R  = df.query(f'hit_id == {hit_left}')[['x','y','z_mod']].values[0]  
        r0 = np.array([x_e, y_e ,z_e])
        
        #"************************************************************************************************************"
        #"***************************Searching CANDIDADATES on the working module*************************************"
        #"************************************************************************************************************"
        # 
        # dz  
        # 
        tmp_candidates = []
        for index, row in df_work_module_window.iterrows(): 
            # Here I only need to have the information of position.    
            r      =  row[['x', 'y', 'z_mod']].values 
            #"************************************************************************************************************"
            #"***************************Searching CANDIDADATES on the working module*************************************"
            #"************************************************************************************************************"    
            ## REFINING  df_work_module_window ############################################################################
            ####################################### dX and dY WINDOW ######################################################
            ###############################################################################################################
            ############################### REMEMBER: work_module, left_mod, mod            ###############################
            ###############################################################################################################
            
            ############################################################################################################### 
            ########################################   NEW WINDOW on X and Y   ############################################ 
            ############################################################################################################### 
            left_cut_x  = x_e - dx 
            right_cut_x = x_e + dx  
            down_cut_y  = y_e - dy 
            up_cut_y    = y_e + dy 
            try : 
                ############################################################################################################ 
                ########################################   DEEP CONDITION on X and Y   ##################################### 
                ############################################################################################################ 
                #print("verifying the kind of x and y ", x, y, type(x), type(y))
                df_work_module_window = df_work_module_window.query(f" {left_cut_x}  < x < {right_cut_x} & {down_cut_y} < y < {up_cut_y} ").copy(deep=True)
                x = r[0] 
                y = r[1] 
                #if  (left_cut_x < x < right_cut_x) and (down_cut_y < y < up_cut_y) :
                #if  (left_cut_x < np.sqrt(x**2 + y**2) < right_cut_x) and (down_cut_y < y < up_cut_y) :  
                
                # the value to change is dr
                
                #WORKING_NOW
                #dr = DR()
                #print("here, I want to print the value of hit module")
                #print(row)
                dx1    = row["dx"]
                dy1    = row["dy"]
                dz1    = row["dz"]
                delta1 = row["delta"]
                dr1    = DR(delta1, dx1, dy1, dz1)
                DR1HISTOGRAM.append(dr1)
                if np.sqrt( (x - x_e)**2 + (y - y_e)**2 ) < dr1 :   
                    """
                    #if len(new_window) > 0 :
                    print("PROOF", part[['x', 'y', 'z']].to_numpy())
                    point = part[['x', 'y', 'z']].to_numpy()
                    plt.plot(z_hits, x_hits) 
                    plt.scatter(z_hits + [ point[2], point[2], point[2]], x_hits+[point[0], left_cut_x, right_cut_x] )
                    plt.xlabel("z")
                    plt.ylabel("x")
                    plt.show() 
                    plt.plot(z_hits, y_hits ) 
                    plt.scatter(z_hits + [ point[2], point[2], point[2]], y_hits+[point[1], down_cut_y, up_cut_y] )
                    plt.xlabel("z")
                    plt.ylabel("y")
                    plt.show()
                    """
                    pass
                else:
                    continue 
            except : 
                print("the new window has a syntax error +++")
                return
            
            hit_id =  row['hit_id']    
            ext_func_value = ext_func(r0, R, r)
            tmp_candidates.append( [hit_id, ext_func_value] ) 
        #"************************************************************************************************************"
        #"***************************In case of not find CANDIDATES on the working module*****************************"
        #"************************************************************************************************************
        if tmp_candidates == [] :                                            
            hit_id_left, hit_id_center, hit_id_right = track[0:3]   
            # the track has its first forwarding 
            if   ( (hit_id_left in left_mod['hit_id'].values ) ): # and  (hit_id_center in mod['hit_id'].values) and (hit_id_right in right_mod['hit_id'].values ) ) : 
                same_track      = track  
                new_tracks.append(same_track)
                continue
            # the track has its second worwarding     
            elif ( (hit_id_left in mod['hit_id'].values )  ):   #and  (hit_center in right_mod['hit_id'].values) ):
                # Add to weak_tracks    
                if(   len(track) == 3 ) :
                    weak_tracks.append(track)
                    continue 
                elif( len(track) >  3 ):
                    same_track   = track  
                    frozen_tracks.append(same_track)
                    continue
        # "************************************************************************************************************" 
        # "************   Choosing new hit_id to complete the track.  *************************************************" 
        # "************************************************************************************************************"     
        df_candidates = pd.DataFrame(tmp_candidates, columns=["hit_id", "ext_fun"])
        if len(tmp_candidates) == 0 : 
            print("an error ocurred with df_candidates")
            return "error"
        new_hit_id    = df_candidates.loc[df_candidates['ext_fun'].idxmin()]['hit_id']
        new_hit_id    = int(new_hit_id)
        # "************************************************************************************************************" 
        # "************   MARKING EACH HIT like "USED" ON THE WORKING MODULE  *****************************************" 
        # "************************************************************************************************************"     
        work_module.loc[ work_module.hit_id == new_hit_id, "used" ] += 1 #True 
        new_track     =  [new_hit_id] + track  
        new_tracks.append(new_track)
    return new_tracks  # this value will be replaced by tracks on the main algorithm


# In[138]:


def doing_plots(df, tracks, name):    
    #global weak_tracks, df, TRACKS
    matplotlib.rc('text', usetex=True)
    matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']

    # List will be used to create a text file 
    # Create plots to show the reconstructed tracks

    #df_real_tracks = df.groupby(['particle_id'])['hit_id'].unique() 

    plt.figure(figsize=(20,10))
    #tracks = tracks.to_list() + weak_tracks 

    for track in tracks :
        # Here I can get the values of the orignal dataframe.
        #data = []
        #for hit in track : # 
        #    data.append(list(df.query(f"hit_id == {hit}").values[0])) # what kind of data we want.
        #data = pd.DataFrame(data, columns=list(df.columns.values) )
        #print("dataframe: ", data[['z', 'y']])
        
        z = df.query(f"hit_id == {track}")['z'].tolist()
        y = df.query(f"hit_id == {track}")['y'].tolist()
        
        plt.plot(z, y, '-', alpha=0.8, lw=2)
        plt.scatter(z, y, marker='+' )
        #print(data['hit_id'])
        #plt.plot(data['z'], data['y'], '-', alpha=0.8, lw=2)
        #plt.scatter(data['z'], data['y'], marker='+' )

        #plt.plot(df['z'], df['y'], '-', alpha=0.8, lw=2, color='C0')
        plt.xlabel(r"\textbf{Z} [mm]")
        plt.ylabel(r'\textbf{Y} [mm]')
        plt.grid(True)
        # tracks.append(data['hit_id'])
    plt.scatter(df['z'], df['y'], marker='+', color='b')
    for particle_id in df.particle_id.unique() : 
        plt.plot(   df.query(f"particle_id =={particle_id}")['z'], df.query(f"particle_id =={particle_id}")['y'], '-', alpha=0.1, lw=1, color='k')
    
    plt.savefig(f"{name}_ZY.png")
    plt.show() 

    plt.figure(figsize=(20,10))
    for track in tracks:
        # Here I can get the values of the orignal dataframe.
        #data = []
        #for hit in track : # 
        #     data.append(list(df.query(f"hit_id == {hit}").values[0])) # what kind of data we want.
        #data = pd.DataFrame(data, columns=list(df.columns.values) )
        #print("dataframe: ", data[['z', 'y']])
        
        z = df.query(f"hit_id == {track}")['z'].tolist()
        x = df.query(f"hit_id == {track}")['x'].tolist()
        
        plt.plot(z, x, '-', alpha=0.8, lw=2)
        plt.scatter(z, x, marker='+' )
        #print(data['hit_id'])
        #plt.plot(data['z'], data['y'], '-', alpha=0.8, lw=2)
        #plt.scatter(data['z'], data['y'], marker='+' )
        
        #plt.plot(df['z'], df['x'], '-', alpha=0.8, lw=2, color='C0')
        plt.xlabel(r"\textbf{Z} [mm]")
        plt.ylabel(r'\textbf{X} [mm]')
        plt.grid(True)
    plt.scatter(df['z'], df['x'], marker='+', color='b' )
    for particle_id in df.particle_id.unique() : 
        plt.plot(   df.query(f"particle_id =={particle_id}")['z'], df.query(f"particle_id =={particle_id}")['x'], '-', alpha=0.2, lw=1, color='k')
    plt.savefig(f"{name}_ZX.png")
    plt.show() 
    
    plt.figure(figsize=(20,10))
    for track in tracks:
        # Here I can get the values of the orignal dataframe.
        #data = []
        #for hit in track : # 
        #     data.append(list(df.query(f"hit_id == {hit}").values[0])) # what kind of data we want.
        #data = pd.DataFrame(data, columns=list(df.columns.values) )
        #print("dataframe: ", data[['z', 'y']])
        
        y = df.query(f"hit_id == {track}")['y'].tolist()
        x = df.query(f"hit_id == {track}")['x'].tolist()
        
        plt.plot(x, y, '-', alpha=0.8, lw=2)
        plt.scatter(x, y, marker='+' )
        #print(data['hit_id'])
        #plt.plot(data['z'], data['y'], '-', alpha=0.8, lw=2)
        #plt.scatter(data['z'], data['y'], marker='+' )
        plt.scatter(df['x'], df['y'], marker='+' )
        #plt.plot(   df['x'], df['y'], '-', alpha=0.8, lw=2, color='C0')
        plt.xlabel(r"\textbf{Y} [mm]")
        plt.ylabel(r'\textbf{X} [mm]')
        plt.grid(True)
        # tracks.append(data['hit_id'])
    
    plt.scatter(df['x'], df['y'], marker='+', color='b')        
    for particle_id in df.particle_id.unique() : 
        plt.plot(   df.query(f"particle_id =={particle_id}")['x'], df.query(f"particle_id =={particle_id}")['y'], '-', alpha=0.2, lw=1, color='k')
    plt.savefig(f"{name}_XY.png")
    plt.show()


# In[139]:


def time_histogram(tracks):
    global df
    plt.figure(figsize=(10, 5))
    time_difference = []
    for track in tracks : 
        for i_hit in range(1, len(track), 1):
            hit1 = track[i_hit-1]
            hit2 = track[i_hit]
            t1 = df.query(f"hit_id == {hit1}")['t_c'].values[0]
            t2 = df.query(f"hit_id == {hit2}")['t_c'].values[0]
            time_difference.append(t2 - t1)  
    plt.hist(time_difference, bins = 100)
    plt.show()

# # MAIN 

# In[145]:

################################### MAIN ###############################################
########################################################################################
############################ GENERAL ALGORITHM #########################################
########################################################################################
########################################################################################

# Probably I can put on this 
def search_by_triplet(NAME="55microns50psInner55microns50psOuter_EventNumber.txt", DPHI=0.01, SIGMA_T=1, SIGMA_Z=1, TIME_RESOLUTION = True, EVENT=1):    
    global theta_scatt
    global dr
    global flagged
    global sigma_z_origin
    global dx, dy
    global frozen_tracks
    global M_i
    global event, df, time_resolution    
    global fraction, df
    global modules, dphi, mod, right_mod, left_mod, sigma_t, work_module, sigma_z, tracks, weak_tracks
    time_resolution = TIME_RESOLUTION 
    
    ### switch time ###...
    if   time_resolution == True:                                             # on all cases where be necessary
        print("the TIME_RESOLUTION is activated ... ")                        # implement timing information here 
    ### switch time ###...
    elif time_resolution == False:                                            # on all cases where be necessary
        print("the TIME_RESOLUTION is de-activated ... ")                     # on all cases where be necessary
    T1 = time.time()  # Timing The Run-Time
    ########################################################################################
    ###############################   PARAMETERS   #########################################
    ########################################################################################
    #suggestion
    # Perhaps for the hit precision a suitable naming scheme is res_x, res_y, res_z, res_t (i.e. RESolution)
    # For the beam spread, if you need variables, you can use PVspread_x, PVspread_y etc 
    
  
    #Spatial and timing coordinates of the PVs are smeared according to a Gaussian distribution with the following widths:
    #sigma(t) = 186ps
    #sigma(z) = 44.7mm
    #sigma(x)= sigma(y)=40microns 0.04 [mm]
    dx       = 0.1              # General Parameters [mm]    where                                                   
    dy       = 0.1              # General Parameters [mm]    where 
    dr       = np.sqrt(2) * dx  # ????? 
    dphi     = DPHI             # The windows is a variable quantity that depends dron phi_ext_base 
    
    # We can assume a mean momentum of 2000 MeV, and since most particles are pions (Mass = 140 MeV) this gives beta = 1
    # theta_0 = [13.6 * 0.085] / [beta * momentum_in_MeV]
    # In the small-angle approximation, tan(theta) = theta, 
    # so this means a '1 sigma' search window of [0.58*z_diff] microns 
    # where d_diff is the module separation in mm - so for 100 mm separation 
    # a '1 sigma' search window would be 58 microns in radius. In reality we might want to use 3 sigma radius so multiply this by three. 
    # Does this make sense?
    
    beta = 1. 
    momentum_in_MeV = 2000 # MeV 
    theta_scatt = 13.6 * 0.085 / (beta * momentum_in_MeV)  #  So this means a typical scattering angle of 13.6*0.085/2000 =c     

    sigma_t  = SIGMA_T             # General Parameters [ps]     where 
    sigma_z  = SIGMA_Z             # General Parameters [mm]     where 
    sigma_z_origin = 3* 44.7       # 50mm               [mm]  where 
    fraction = 1                   # General Parameters          where 
    event    = EVENT               # General Parameters          where    
    flagged  = 1                 # General Parameters integer number  
    
    #phi_window =           # phi_extrapolation_base + np.abs( hit_Zs[h_center]) * phi_extrapolation_coef
    #phi_extrapolation_coef = 0.02 
    #phi_extrapolation_base = 0.03 
    m = 24                  # number of modules counted from the left. from 1 to 24. No more. 
    ########################################################################################
    ########################################################################################
    ######################################################################################## 
    
    new_tracks    = []                   # where data is unmodified.   
    frozen_tracks = []                   # these tracks are formed by more than 4 hits, it is important that, it will be joined with the frozen_tracks 
    tracks        = []                   # 
    weak_tracks   = []                   # 
    df = reading_data(NAME, fraction, event)   # where data is unmodified.
    
    # *********************IMPORTANT********************************************************
    # The information of tracks is ordered. 
    # Because, each of its elements are an ordered list according to module layers.
    # However, the information of hits are unique and not matter if they are a ordered set. 
    # But it was filled out in order

    # SEPARATION BY MODULE  
    modules = sortbyphi()                    # this line modify df adding the z_correct
    # print("second_error", len(modules[0]))   
    #for i in range(len(modules)):             
    #    print(modules[i]['t_c'])            # 
    # FIND CANDIDATE WINDOWS. In order to minimize the amount of candidates considered in subqsequent steps.
    ################################################    
    ###### Ordering Modules Accordig to Phi: #######
    ################################################
    print("ordering modules accordig to phi ... ")
    for M_i in range(len(modules)-1-1, len(modules)-m-2, -1): 
        #M_i = M_i - 1
        left_mod     =  modules[M_i - 1] #.copy(deep=True)      
        mod          =  modules[M_i    ] #.copy(deep=True)   
        right_mod    =  modules[M_i + 1] #.copy(deep=True)  
        modules[M_i] =  findcandidatewindows(left_mod, mod, right_mod).copy(deep=True)

    ######PRINCIPAL LOOP OVER MODULES#################################################
    ##################################################################################   
    ##################################################################################   
    ############# ITERATION OVER MODULES ( ):#########################################
    for M_i in range(len(modules)-1-1, len(modules)-m-2, -1) :  # the number two is due to 1. index postion default. 2. 
        t1 = time.time()    # TIMING THE RUNNING OVER A MODULE             
        print(f"module number {M_i}")
        #M_i = M_i - 1
        #1th STEP:  assigning NOTATION
        left_mod  =  modules[M_i - 1]#.copy(deep=True)   
        mod       =  modules[M_i    ]#.copy(deep=True)   
        right_mod =  modules[M_i + 1]#.#copy(deep=True)

        new_seeds = trackseeding() 
        
        name = f"center_module_{M_i}"
        #doing_plots(df, new_seeds,name)
        #"""
        #Adding new seeds to tracks 
        tracks    = tracks + new_seeds 
        # REASIGNING VALUES    
        modules[M_i - 1] = left_mod.copy( deep=True)           
        modules[M_i    ] = mod.copy(      deep=True)              
        modules[M_i + 1] = right_mod.copy(deep=True)        

        # Defining a new module.  
        work_module      = modules[M_i - 2].copy(deep=True) 
        
        new_tracks       = track_forwarding()         
        tracks           = new_tracks
        name = f"tracks_at_step_{M_i}"
        #doing_plots(df, tracks,name)
        
        modules[M_i - 2] = work_module.copy(deep=True)
        
        t2 = time.time()
        print("time per module", t2-t1)  
    
    print("FINDING TRACKS FINISHED") 
    T2 = time.time()# Timing The Run-Time
    print("RUN TOTAL TIME PER EVENT IS : ", T2-T1) 
    
    df_real_tracks = df.groupby(['particle_id'])['hit_id'].unique()  
    """
    #df_real_tracks = df_real_tracks[df_real_tracks.apply(len) > 2]
    if len(tracks) > 0 :
        print("********************************************************")
        print("SCORING using tracks")              
        print(Scoring(df_real_tracks, tracks))
        time_histogram(tracks)
    if len(weak_tracks) > 0 :  
        print("********************************************************")
        print("SCORING using weak_tracks")
        print(Scoring(df_real_tracks, weak_tracks))
        time_histogram(weak_tracks)
    if len(frozen_tracks) > 0 : 
        print("********************************************************")
        print("SCORING using frozen_tracks")  
        print(Scoring(df_real_tracks, frozen_tracks))
        time_histogram(frozen_tracks)
    if len( tracks + frozen_tracks + weak_tracks ) > 0 : 
        print("********************************************************")
        print("SCORING using all tracks")  
        print(Scoring(df_real_tracks, tracks + frozen_tracks + weak_tracks))
        time_histogram(tracks + frozen_tracks + weak_tracks)
    else :
        print("********************************************************")
        print("Absolutely no track is founded: ")
    """
    #doing_plots(df, tracks+frozen_tracks+weak_tracks, name="proof")
    return Scoring(df_real_tracks, tracks + frozen_tracks)
# search_by_triplet_algorithm.ipynb


# In[146]:


#search_by_triplet(True)


# In[142]:


#plt.hist(GRALDR1)
#plt.show()


# In[143]:


#plt.hist(GRALDR2)
#plt.show()


# # Delete all found Real tracks. And See only those that are Fake Tracks

# In[144]:

"""
def FilteringTracks(real_tracks, tracks):
    
    return fake_tracks
"""

# # Implement an algorithm.