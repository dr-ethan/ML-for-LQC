import numpy as np
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd




test_filepath = "testfile.txt"
two_point_identifiers = ['2pt_D_gold_msml5_fine.ll','2pt_D_nongold_msml5_fine.ll','2pt_msml5_fine_K_zeromom.ll']
localvar_identifiers = ['localtempvec_pmax_3pt_T16_msml5_fine.ll', 'localtempvec_pmax_3pt_T19_msml5_fine.ll','localtempvec_pmax_3pt_T22_msml5_fine.ll','localtempvec_pmax_3pt_T25_msml5_fine.ll']

correlator_variable = two_point_identifiers[1] #this is the two point dataset, change from 0-2 for different datasets
identifiers =[correlator_variable,localvar_identifiers[0], localvar_identifiers[1],localvar_identifiers[2],localvar_identifiers[3]] #big list of all the identifers



def dict_float(filepath): 
    data_dict ={}
    current_dataset = None
    with open(filepath, 'r') as file:
        for line in file:
            '''
            This part of the code creates a list in the dictionary for each identifer 
            identifier(s) - The amount of different experiments in the file, for 2pt-3pt-qsqmax-scalar, it is 7 unique experiments
            '''
            for identifier in identifiers: 
                if line.startswith(identifier):
                    current_dataset=identifier 
                    if current_dataset not in data_dict:
                        data_dict[current_dataset] = []   
            '''
            this part of the code turns the file data type from string to float
            it also removes the identifier in the .txt file and the first uncorrelated number in 2pt so it is purely float
            '''
            if current_dataset != None:
                if current_dataset in two_point_identifiers: data_points = line.split()[2:] #ignores the random first number and identifier for 2pt data
                else: data_points = line.split()[1:]  #ignores the name of the list
                data_points = [float(dp) for dp in data_points]
                if current_dataset in localvar_identifiers:
                    data_points = localvar_strip(data_points,current_dataset)
                data_points = np.abs(data_points)
                data_points = np.log(data_points) #make a linear relationship for correlation calculation later
                data_dict[current_dataset].append(data_points)
                
    file.close()
    return data_dict

def localvar_strip(data,current_identifier):
    '''
    cuts off data past 3pt element limit by using the order of 3pt tags in the localvar list
    '''
    for identifier in localvar_identifiers:
        if current_identifier == identifier: data = data[:(16+(3*localvar_identifiers.index(identifier)))] 
    return data

def plot_heatmap(data_corr_dict):
    '''
    this function shows a heatmap figure for each of the localtempvec identifiers (T16, T19, T22, T25)
    
    data_corr_dict is a dictionary containing 2D arrays of the correlation variables for the respective identifiers
    '''
    for identifier,data in data_corr_dict.items():
        data = np.array(data)
        data = np.abs(data)
        limit = 16+(3*localvar_identifiers.index(identifier))
        time = np.linspace(1,limit,limit, dtype=int)
        plt.figure(figsize=(8,8))
        sns.heatmap(data,annot=False, cmap='plasma', square=True,xticklabels=time,yticklabels=time)
        plt.title('Correlation Grid Heatmap with 'f'{correlator_variable} and 'f'{identifier}')
        plt.xlabel('Time Slices for'f'{correlator_variable}')
        plt.ylabel('Time Slices for'f'{identifier}')
        plt.show()
    return

def pearson_r_calculator(x,y,limit,max_config):
    '''
    the function for finding correlation coefficient
    x is 3pt data, y is 2pt data
    
    this function calculates the pearsonr coefficient for one column in 2pt to all other columns in 3pt
    E.G. for T16 there would be 16^2 coefficients which is plotted in heatmap later
    '''
    r = np.zeros(shape=(limit,limit))
    
    for i in range(limit):
        for j in range(limit):
            x_mean = np.mean(x[:max_config,j]) 
            y_mean = np.mean(y[:max_config,i]) #set the means for the column j
            x_column = x[:max_config,j]
            y_column = y[:max_config,i] #set the columns for the column j
            numerator = np.sum((x_column-x_mean)*(y_column-y_mean)) #first half of the calculation
            denominator = np.sqrt(np.sum((x_column-x_mean)**2)*np.sum((y_column-y_mean)**2)) #second half of the calculation
            r[i,j] = numerator/denominator
    return r

def correlation(data_dict,correlator):
    '''
    the function that does the calculation for the pearson correlation. 
    sorts the dictionary by only selecting one identifier's data at a time, then computes them in pearson_r_calculator
    limit is number of elements in the 3 pt data, e.g. T16 has limit = 16
    
    max_config is the maximum amount of configurations allowed. 
    e.g. if T22 has 420 configs and 2pt has 490 configs, then max_config = 420 so both datasets have same shape.
    (not a problem when receive raw data)
    '''
    correlator_variable_data = np.array(data_dict[correlator]) #puts the 2pt data from dictionary into a new array
    correlation_dict = {} #for combining later
    for identifier, data in data_dict.items(): #for each dataset
        if identifier is not correlator: #to make sure it doesnt correlate its own data
            to_correlate_data=np.array(data)
            limit = 16+(3*localvar_identifiers.index(identifier)) 
            max_configurations = min(correlator_variable_data.shape[0],to_correlate_data.shape[0]) 
            pearson_r_values = pearson_r_calculator(to_correlate_data,correlator_variable_data,limit,max_configurations)
            correlation_dict[identifier] = pearson_r_values
    return correlation_dict

filepath = "2pt-3pt-qsqmax-scalar.gpl"
data_float_dict = dict_float(filepath)
correlation_library = correlation(data_float_dict,correlator_variable)
plot_heatmap(correlation_library)


'''
heatmap_data = np.array([corr for corr in correlation_library.values()])
heatmap_data = abs(heatmap_data)

plt.figure(figsize=(8, 8))
sns.heatmap(heatmap_data, annot=False, cmap='plasma', square=True,xticklabels=[f'{i+1}' for i in range(32)], yticklabels=list(correlation_library.keys()))
plt.title('Correlation Heatmap with 'f'{correlator_variable}')
plt.xlabel('Columns')
plt.ylabel('')
plt.show()
'''