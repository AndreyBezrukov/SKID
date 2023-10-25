# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 18:35:01 2023

@author: Andrey.Bezrukov
"""
developer_contact = 'Andrey.Bezrukov@ul.ie'
software_version = '1.0'

print('################################################################################################')
print('      Sorption kinetics isotherm determination, SKID, V{0}'.format(software_version))
print('      Author: Andrey A. Bezrukov')
print('################################################################################################')

import time
from datetime import datetime
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import interpolate
from sklearn.cluster import DBSCAN
import numpy as np

import PySimpleGUI as sg
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
'''
with warnings.catch_warnings():
    warnings.simplefilter("ignore") 
'''
plt.rcParams['toolbar'] = 'toolmanager'
from matplotlib.backend_tools import ToolBase, ToolToggleBase

### declare custom exceptions
class CannotReadFile(Exception):
    pass
class WrongFileFormat(Exception):
    pass

##################################################
####### define  functions 
##################################################

class file_Sorption:
    
    def read_file(self, path, filename):
        #print(path +'/'+ filename)        
        try:
            df = pd.read_excel(path + filename, sheet_name='DVS Data')
            if df.columns[0] == 'DVS-INTRINSIC DATA FILE':
                self.filename = filename
                self.instrument = df.columns[0].split()[0]
                self.sample_mass = df.iloc[4, 7]
                self.fluid = 'water vapor'
                comments  = str(df.iloc[3, 1])
                method = str(df.iloc[1, 1])
                df.columns = df.iloc[22, :]
                df = df.iloc[23:, :]
                df = df.reset_index(drop=True)
                df = df.astype('float')
                # renaming columns to common notation
                df = df[['Time (min)', 'dm (%) - ref', 'Target RH (%)', 'Actual RH (%)', 'Target Sample Temp', 'Actual Sample Temp']]
                df.columns = ['time', 'uptake', 'RH_target', 'RH_actual', 'temp_target', 'temp_actual']
                self.temperature = 'Actual Sample Temp: {0:.2f} +- {1:.2f}'.format(df[df.temp_target==df.temp_target.min()].temp_actual.mean(), df[df.temp_target==df.temp_target.min()].temp_actual.std())
                self.equilibration_interval = '---'
                self.comments = comments
                self.method = method
                self.data = df
                #print(self.data.head())
            elif df.columns[0] == 'DVS-Advantage-Plus-Data-File':
                self.filename = filename
                self.instrument = df.columns[0][:-10]
                self.sample_mass = df.iloc[27, 1]
                self.fluid = 'water vapor'
                comments  = str(df.iloc[8, 1])
                method = str(df.iloc[3, 1])
                df.columns = df.iloc[39, :]
                df = df.iloc[40:, :30]
                df = df.reset_index(drop=True)
                df = df[[i for i in df.columns if (i == i) & (i != 'Chiller State')]]
                df = df.astype('float')
                # renaming columns to common notation
                df = df[['Time [minutes]', 'dm (%) - ref', 'Mass [mg]', 'Target Partial Pressure (Solvent A) [%]', 'Measured Partial Pressure (Solvent A) [%]', 'Target Preheater Temp. [celsius]', 'Measured Preheater Temp. [celsius]']]
                df.columns = ['time', 'uptake', 'mass', 'RH_target', 'RH_actual', 'temp_target', 'temp_actual']
                self.temperature = 'Temp. [celsius]: {0:.2f} +- {1:.2f}'.format(df[df.temp_target==df.temp_target.min()].temp_actual.mean(), df[df.temp_target==df.temp_target.min()].temp_actual.std())
                self.equilibration_interval = '---'
                self.comments = comments
                self.method = method
                self.data = df
                #print(self.data.head())
        except Exception as e:
            print(datetime.now())
            print(e)
        #print(self.__dict__.keys())
        # test if import was correct
        if (list(self.__dict__.keys()) == ['filename', 'instrument', 'sample_mass', 'fluid', 'temperature', 'equilibration_interval', 'comments', 'method', 'data'])  :
            print(datetime.now())
            print('File {0} read succesfull'.format(filename))
            self.import_success = True
        else:
            self.import_success = False
            
        

##################################################
####### run analysis 
##################################################


###################  Start GUI   ############################

sg.change_look_and_feel('DefaultNoMoreNagging')     

layout1 =   [   
            [sg.Text('Figure settings:   font size', size=(19, 1)), sg.In(default_text='16', size=(5, 1), key= 'text_size'), sg.Text('    dpi', size=(5, 1)), sg.In(default_text='100', size=(5, 1), key= 'dpi'),],
            [sg.Text('STEP 1: read kinetics data', auto_size_text=False, justification='left',  font=("Helvetica", 12, "bold"))],
            [sg.Text('Read kinetics data for isotherm determination:', auto_size_text=False, justification='left')], 
            [sg.InputText('Choose file', key='kinetics_file'), sg.FilesBrowse()],
            [sg.Button('Open')],
            [sg.Text('STEP 2: determine isotherm', auto_size_text=False, justification='left',  font=("Helvetica", 12, "bold"))], 
            [sg.Text('Select calculation parameters:', auto_size_text=False, justification='left')], 
            [sg.Text('cycle number', size=(30, 1)), sg.In(default_text='1', size=(5, 1), key= 'cycle_number')],
            [sg.Text('first derivative moving average window', size=(30, 1)), sg.In(default_text='50', size=(5, 1), key= 'window1')],
            [sg.Text('second derivative moving average window', size=(30, 1)), sg.In(default_text='10', size=(5, 1), key= 'window2')],
            [sg.Text('number of iterations for k determination', size=(30, 1)), sg.In(default_text='10', size=(5, 1), key= 'k_iterations')],
            [sg.Button('Calculate isotherm') ],
            [sg.Text('STEP 3: save result', auto_size_text=False, justification='left',  font=("Helvetica", 12, "bold"))],
            [sg.Text('Save calculated isotherm:', auto_size_text=False, justification='left')], 
            [sg.Text('filename:', size=(7, 1)), sg.In(default_text='isotherm.csv', size=(50, 1), key= 'isotherm_file')],
            [sg.Button('Save')],
            ]

kinetics_uploaded = False
isotherm_calculated = False

window = sg.Window('Sorption kinetics isotherm determination, SKID, V'+software_version, layout1, default_element_size=(40, 1), grab_anywhere=False)    
while True:
    event, values = window.read() 
    
    ################# window closed ################
    if event == sg.WIN_CLOSED:
        break
    
    
    ########### test input values ############
    input_values_valid = True

    def value_test_int(v, message):
        try: 
            if int(v)<=0:
                print(datetime.now())
                print('Error:', message)
                sg.popup('Error',  message)
                return False
            else: return True
        except:
            print(datetime.now())
            print('Error:', message)
            sg.popup('Error',  message)
            return False
    def input_values_test():
        return all([value_test_int(values['cycle_number'],  '\'cycle number\' parameter MUST be positive nonzero integer'),
                              value_test_int(values['window1'],  '\'first derivative moving average window\' parameter MUST be positive nonzero integer'),
                              value_test_int(values['window2'],  '\'second derivative moving average window\' parameter MUST be positive nonzero integer'),
                              value_test_int(values['k_iterations'],  '\'number of iterations for k determination\' parameter MUST be positive nonzero integer'),
                              value_test_int(values['text_size'],  'Figure \'font size\' parameter MUST be positive nonzero integer'),
                              value_test_int(values['dpi'],  'Figure \'dpi\' parameter MUST be positive nonzero integer'),
                              ])
    
    ################# Import kinetics data ################
    if event == 'Open': 
        if not input_values_test():
            continue
        if values['kinetics_file']=='Choose file':
            print('Choose file')
            sg.popup('Choose file')
            continue
        path = values['kinetics_file'][:-len(values['kinetics_file'].split('/')[-1])]
        filename = values['kinetics_file'].split('/')[-1]
        print(datetime.now())
        print('Reading file:', filename)
        ## import kinetics
        Sorption_kinetics = file_Sorption()
        try:
            Sorption_kinetics.read_file(path, filename)
            if Sorption_kinetics.import_success:
                if Sorption_kinetics.fluid != 'water vapor':
                    del Sorption_kinetics
                    print('Wrong fluid: ',Sorption_kinetics.fluid)
            else:
                del Sorption_kinetics
                print(datetime.now())
                print(filename, 'failed')
               #continue
        except Exception as e:
            print(datetime.now())
            print(e)
            print(filename, 'failed')
           #continue

        kinetics_uploaded = True
        
        fig, ax = plt.subplots(figsize=(10, 9), dpi=int(values['dpi']))
        ax.plot(Sorption_kinetics.data['time'], Sorption_kinetics.data['uptake'], c='g')
        ax2 = ax.twinx()
        ax2.plot(Sorption_kinetics.data['time'], Sorption_kinetics.data['RH_actual'], c='b')
        #ax2.plot(Sorption_kinetics.data['time'], Sorption_kinetics.data['temp_actual'], c='r')
        ax.set_xlabel('time, min', fontsize=values['text_size'])
        ax.set_ylabel('uptake, wt.%', fontsize=values['text_size'], c='g')
        ax.tick_params(axis='x', labelsize=values['text_size'])
        ax.tick_params(axis='y', labelsize=values['text_size'])
        ax2.set_ylabel('RH actual, %', fontsize=values['text_size'], c='b')
        ax2.tick_params(axis='x', labelsize=values['text_size'])
        ax2.tick_params(axis='y', labelsize=values['text_size'])
        fig.suptitle(''.join([filename, ', ', str(Sorption_kinetics.sample_mass), ' mg sample' ]),
                             fontsize=values['text_size'])
        plt.show(block=False)

        Sorption_kinetics.data['cycle_split'] = Sorption_kinetics.data['RH_target'].diff().fillna(0)
        Sorption_kinetics.data['cycle_split_temp'] = Sorption_kinetics.data['temp_target'].diff().fillna(0)
        split_index_ads = Sorption_kinetics.data.index[(Sorption_kinetics.data['cycle_split']>0)].to_list()
        split_index_des = Sorption_kinetics.data.index[(Sorption_kinetics.data['cycle_split']<0)].to_list()
        split_index_temp = Sorption_kinetics.data.index[(Sorption_kinetics.data['cycle_split_temp']!=0)].to_list()
    
    ################# Calculate isotherm ################
    if (event == 'Calculate isotherm')&(kinetics_uploaded!=True): 
        print(datetime.now())
        print('Error, open kinetics (step 1) before calculating isotherm')
        sg.popup('Error, open kinetics (step 1) before calculating isotherm')
    if (event == 'Calculate isotherm')&(kinetics_uploaded==True):
        if not input_values_test():
            continue
        window1 = int(values['window1'])
        window2 = int(values['window2'])
        path = values['kinetics_file'][:-len(values['kinetics_file'].split('/')[-1])]
        filename = values['kinetics_file'].split('/')[-1]
        cycle_number = int(values['cycle_number'])-1
        
        if (cycle_number>len(split_index_ads))|(cycle_number>len(split_index_des)):
            print(datetime.now())
            print('Error, \'cycle number\' parameter exceeds max: {0}'.format(min(len(split_index_ads), len(split_index_des) )))
            sg.popup('Error, \'cycle number\' parameter exceeds max: {0}'.format(min(len(split_index_ads), len(split_index_des) )))
            continue
        
        print(datetime.now())
        print('Isotherm calculation parameters:\ncycle number={0}\nfirst derivative moving average window={1}\nsecond derivative moving average window={2}\nnumber of iterations for k determination={3}'.format(cycle_number+1, 
                                                                                                                                                                                                                        window1, 
                                                                                                                                                                                                                        window2, 
                                                                                                                                                                                                                        values['k_iterations'],
                                                                                                                                                                                                                        ))
        
        fig, ax = plt.subplot_mosaic(    
        """
        AABB
        .HH.
        """
                                  ,figsize=(10, 9), constrained_layout=True, dpi=int(values['dpi']))
        fig.suptitle(''.join([filename, ', ', str(Sorption_kinetics.sample_mass), ' mg sample' ]),
                             fontsize=values['text_size'])
        
        # uptake adsorption
        try:
            w_ads = pd.DataFrame({'time':Sorption_kinetics.data.iloc[split_index_ads[cycle_number]:split_index_des[cycle_number], :]['time'] - Sorption_kinetics.data.iloc[split_index_ads[cycle_number]:split_index_des[cycle_number], :]['time'].min(), 
                                  'uptake':Sorption_kinetics.data.iloc[split_index_ads[cycle_number]:split_index_des[cycle_number], :]['uptake'],
                                  'RH_actual':Sorption_kinetics.data.iloc[split_index_ads[cycle_number]:split_index_des[cycle_number], :]['RH_actual']
                                  })
        except Exception as e:
            print(datetime.now())
            print(e)
           #continue
        ax['A'].plot(w_ads['time'],
                      w_ads['uptake'],
                      c='g',
                      label=''.join([filename, ', ', str(Sorption_kinetics.sample_mass), ' mg' ])
                      )
        #ax['A'].legend()
        ax['A'].set_xlabel('time, min', fontsize=values['text_size'])
        ax['A'].set_ylabel('uptake, wt.%', fontsize=values['text_size'], c='g')
        ax['A'].tick_params(axis='x', labelsize=values['text_size'])
        ax['A'].tick_params(axis='y', labelsize=values['text_size'])
        ax['A'].set_title('Adsorption kinetics', fontsize=values['text_size'])
        ax2 = ax['A'].twinx()
        ax2.plot(w_ads['time'],
                 w_ads['RH_actual'], 
                 c='b'
                 )
        ax2.set_ylabel('RH actual, %', fontsize=values['text_size'], c='b')
        ax2.tick_params(axis='y', labelsize=values['text_size'])
        
        # first derivative adsorption
        first_derivative_dwdt_ads = pd.DataFrame({'dwdt':(w_ads['uptake'].diff()/w_ads['time'].diff()).rolling(window=window1, min_periods=1, center=True).mean(),
                                                  'uptake':w_ads['uptake']
                                                  })

        
        # first derivative adsorption where second derivative is negatinve
        first_derivative_dwdt_dw_ads = pd.DataFrame({'dwdt_dw':(first_derivative_dwdt_ads['dwdt'].diff()/first_derivative_dwdt_ads['uptake'].diff()).rolling(window=window2, min_periods=1, center=True).mean(),
                                                     'uptake':first_derivative_dwdt_ads['uptake']
                                                     })
        def normalize(x):
            return (x-x.min())/(x.max()-x.min())
        X = np.array([[i[0], i[1]] for i in zip(normalize(first_derivative_dwdt_ads[first_derivative_dwdt_dw_ads['dwdt_dw']<0]['uptake']), normalize(first_derivative_dwdt_ads[first_derivative_dwdt_dw_ads['dwdt_dw']<0]['dwdt']))  ])
        clustering_ads = DBSCAN(eps=0.1, min_samples=1).fit(X)
        #print(clustering_ads.labels_)

        
        # uptake desorption
        try:
            temp_index = [split_index_des[cycle_number], split_index_ads[cycle_number+1]]
        except: 
            temp_index = [split_index_des[cycle_number], Sorption_kinetics.data.index.max()]
        if [i for i in split_index_temp if (i>temp_index[0])&(i<temp_index[1])] != []:
            temp_index = [temp_index[0], min(min([i for i in split_index_temp if (i>temp_index[0])&(i<temp_index[1])]), temp_index[1])]
        try:
            w_des = pd.DataFrame({'time':Sorption_kinetics.data.iloc[temp_index[0]:temp_index[1], :]['time'] - Sorption_kinetics.data.iloc[temp_index[0]:temp_index[1], :]['time'].min(), 
                                  'uptake':Sorption_kinetics.data.iloc[temp_index[0]:temp_index[1], :]['uptake'], 
                                  'RH_actual':Sorption_kinetics.data.iloc[temp_index[0]:temp_index[1], :]['RH_actual'], 
                                  })
        except Exception as e:
            print(datetime.now())
            print(e)
           #continue      
        ax['B'].plot(w_des['time'],
                      w_des['uptake'],
                      c='g',
                      label=''.join([filename, ', ', str(Sorption_kinetics.sample_mass), ' mg' ])
                      )
        #ax['B'].legend()
        ax['B'].set_xlabel('time, min', fontsize=values['text_size'])
        ax['B'].set_ylabel('uptake, wt.%', fontsize=values['text_size'], c='g')
        ax['B'].tick_params(axis='x', labelsize=values['text_size'])
        ax['B'].tick_params(axis='y', labelsize=values['text_size'])
        ax['B'].set_title('Desorption kinetics', fontsize=values['text_size'])
        ax3 = ax['B'].twinx()
        ax3.plot(w_des['time'],
                 w_des['RH_actual'], 
                 c='b'
                 )
        ax3.set_ylabel('RH actual, %', fontsize=values['text_size'], c='b')
        ax3.tick_params(axis='y', labelsize=values['text_size'])
        
        # first derivative adsorption
        first_derivative_dwdt_des = pd.DataFrame({'dwdt':(w_des['uptake'].diff()/w_des['time'].diff()).rolling(window=window1, min_periods=1, center=True).mean(),
                                                  'uptake':w_des['uptake']
                                                  })

        
        # first derivative desorption where second derivative is negatinve
        first_derivative_dwdt_dw_des = pd.DataFrame({'dwdt_dw':(first_derivative_dwdt_des['dwdt'].diff()/first_derivative_dwdt_des['uptake'].diff()).rolling(window=window2, min_periods=1, center=True).mean(),
                                                     'uptake':first_derivative_dwdt_des['uptake']
                                                     })
        def normalize(x):
            return (x-x.min())/(x.max()-x.min())
        X = np.array([[i[0], i[1]] for i in zip(normalize(first_derivative_dwdt_des[first_derivative_dwdt_dw_des['dwdt_dw']<0]['uptake']), normalize(first_derivative_dwdt_des[first_derivative_dwdt_dw_des['dwdt_dw']<0]['dwdt']))  ])
        clustering_des = DBSCAN(eps=0.1, min_samples=1).fit(X)
        #print(clustering_des.labels_)

        
        ### perform RH calibration
        ### use DVS adventure k for TGA pan = 0.3
        k_adventure_TGA_min = 0.01
        k_adventure_TGA_max = 5
        
        try:
            
            f_des = interpolate.interp1d(first_derivative_dwdt_des['uptake'],  first_derivative_dwdt_des['dwdt'], fill_value="extrapolate")
            def interpolate_first_derivative_dwdt_dw_des(t):
                return f_des(t)
            f_ads = interpolate.interp1d(first_derivative_dwdt_ads['uptake'],  first_derivative_dwdt_ads['dwdt'], fill_value="extrapolate")
            def interpolate_first_derivative_dwdt_dw_ads(t):
                return f_ads(t)
            
            first_derivative_dwdt_dw_intersection = [first_derivative_dwdt_ads['uptake'][first_derivative_dwdt_dw_ads['dwdt_dw']<0][clustering_ads.labels_==np.bincount(clustering_ads.labels_).argmax()].min(),
                                                    first_derivative_dwdt_des['uptake'][first_derivative_dwdt_dw_des['dwdt_dw']<0][clustering_des.labels_==np.bincount(clustering_des.labels_).argmax()].max()]
            #print('Intersection: ',first_derivative_dwdt_dw_intersection)
            desdata_interpolated = interpolate_first_derivative_dwdt_dw_des(first_derivative_dwdt_ads[(first_derivative_dwdt_ads.uptake>first_derivative_dwdt_dw_intersection[0])&(first_derivative_dwdt_ads.uptake<first_derivative_dwdt_dw_intersection[1])]['uptake'])
            adsdata_interpolated = interpolate_first_derivative_dwdt_dw_ads(first_derivative_dwdt_ads[(first_derivative_dwdt_ads.uptake>first_derivative_dwdt_dw_intersection[0])&(first_derivative_dwdt_ads.uptake<first_derivative_dwdt_dw_intersection[1])]['uptake'])
            RH_max = Sorption_kinetics.data.iloc[split_index_ads[cycle_number]:split_index_des[cycle_number], :]['RH_target'].max()
            
            for attempt in range(int(values['k_iterations'])):
                k_adventure_TGA = (k_adventure_TGA_min+k_adventure_TGA_max)/2
                #print(k_adventure_TGA)
                if ((-desdata_interpolated/k_adventure_TGA*Sorption_kinetics.sample_mass)<(RH_max - adsdata_interpolated/k_adventure_TGA*Sorption_kinetics.sample_mass)).all():
                    k_adventure_TGA_max = k_adventure_TGA
                else: 
                    k_adventure_TGA_min = k_adventure_TGA
        
            print('Fitted parameter k: ',k_adventure_TGA)
            
            first_derivative_dwdt_ads['dwdt_scaled'] = RH_max - first_derivative_dwdt_ads['dwdt']/k_adventure_TGA*Sorption_kinetics.sample_mass
            first_derivative_dwdt_des['dwdt_scaled'] = -first_derivative_dwdt_des['dwdt']/k_adventure_TGA*Sorption_kinetics.sample_mass
        
            ax['H'].scatter(first_derivative_dwdt_ads[first_derivative_dwdt_dw_ads['dwdt_dw']<0][clustering_ads.labels_==np.bincount(clustering_ads.labels_).argmax()]['dwdt_scaled'],
                             first_derivative_dwdt_ads[first_derivative_dwdt_dw_ads['dwdt_dw']<0][clustering_ads.labels_==np.bincount(clustering_ads.labels_).argmax()]['uptake'],
                         #c='b',
                         label='adsorption',
                         s=5)
            ax['H'].scatter(first_derivative_dwdt_des[first_derivative_dwdt_dw_des['dwdt_dw']<0][clustering_des.labels_==np.bincount(clustering_des.labels_).argmax()]['dwdt_scaled'],
                             first_derivative_dwdt_des[first_derivative_dwdt_dw_des['dwdt_dw']<0][clustering_des.labels_==np.bincount(clustering_des.labels_).argmax()]['uptake'],
                         #c='b',
                         label='desorption',
                         s=5)
            ax['H'].set_xlabel('RH, %', fontsize=values['text_size'])
            ax['H'].set_ylabel('uptake, wt.%', fontsize=values['text_size'])
            ax['H'].tick_params(axis='x', labelsize=values['text_size'])
            ax['H'].tick_params(axis='y', labelsize=values['text_size'])
            ax['H'].set_title('Isotherm determined from kinetics', fontsize=values['text_size'])
            ax['H'].legend()  
            
            df_export_ads = pd.DataFrame(data={'adsorption_RH': first_derivative_dwdt_ads[first_derivative_dwdt_dw_ads['dwdt_dw']<0][clustering_ads.labels_==np.bincount(clustering_ads.labels_).argmax()]['dwdt_scaled'],
                                           'adsorption_uptake': first_derivative_dwdt_ads[first_derivative_dwdt_dw_ads['dwdt_dw']<0][clustering_ads.labels_==np.bincount(clustering_ads.labels_).argmax()]['uptake'],
                                           }).reset_index(drop=True) 
            df_export_des = pd.DataFrame(data={'desorption_RH': first_derivative_dwdt_des[first_derivative_dwdt_dw_des['dwdt_dw']<0][clustering_des.labels_==np.bincount(clustering_des.labels_).argmax()]['dwdt_scaled'],
                                           'desorption_uptake': first_derivative_dwdt_des[first_derivative_dwdt_dw_des['dwdt_dw']<0][clustering_des.labels_==np.bincount(clustering_des.labels_).argmax()]['uptake'],
                                            }).reset_index(drop=True) 
            
            df_export = pd.concat([df_export_ads, df_export_des], axis=1,)
            print(datetime.now())
            print('Isotherm calculation successfull')
            isotherm_calculated = True
  
        except Exception as e: 
            print(datetime.now())
            print(e)
            pass
        
        
        fig.tight_layout(#rect=[0, 0, 1, 0.97])
                         )
        plt.show(block=False)
    
    ################# Export isotherm ################
    if (event == 'Save')&(isotherm_calculated!=True):     
        print(datetime.now())
        print('Error, calculate isotherm (step 2) before saving it')
        sg.popup('Error, calculate isotherm (step 2) before saving it')
    if (event == 'Save')&(isotherm_calculated==True):
        df_export.to_csv(path+values['isotherm_file'], index=False )
        print(datetime.now())
        print('Isotherm saved to: ', path, values['isotherm_file'] )
    


