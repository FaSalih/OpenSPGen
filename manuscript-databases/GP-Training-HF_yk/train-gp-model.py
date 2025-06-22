# -*- coding: utf-8 -*-
"""
Script to train a GP on sigma profiles to predict a physicochemical property.
Given 1 data split (k-fold), and looping over multiple target properties and
datasets.

Sections:
    . Imports
    . Configuration
    . Auxiliary Functions
        . normalize()
        . buildGP()
        . gpPredict()
        . extract_data()
        . find_zero_cols()
        . get_perf_edges()
    . Main Script
    . Plots

Last edit: 2025-01-10
Author: Fathya Salih, Dinis Abranches
"""

# =============================================================================
# Imports
# =============================================================================

# General
import os
import warnings
import time
import sys

# Specific
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
import gpflow
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
from matplotlib import rc
from tqdm import tqdm
import pickle

# =============================================================================
# Configuration
# =============================================================================

# Define normalization methods
featureNorm=None # None,Standardization,MinMax,LogStand,Log+bStand
labelNorm='Standardization' # None,Standardization,MinMax,LogStand,Log+bStand
# GP Configuration
gpConfig={'kernel':'RBF',
          'useWhiteKernel':True,
          'trainLikelihood':True,
          'alpha':10**-2}

# list datasets for training and plotting
datasets = ['sp_hf_svp_yk', 'sp_mullins_vt-2005', 'sp_mullins_no_av',]
dataset_names = ['HF/def2-SVP-YK (No Avg.)', 'Mullins', 'Mullins - No Averaging']

# Get target variable/property name, dataset codes, and display units
varNames=['Molar Mass', 'Boiling Point', 'Density at 20°C', 'RI at 20°C', '$S_{aq}$ at 25°C (g/kg)', 'Vapor Pressure']
codes = ['MM', 'BP', 'D_20', 'RI_20', 'S_25', 'VP']
varUnits=['g/mol', '°C', 'g/cm³', 'RI', 'g/kg', 'Pa']

# define results and dataset folders
dataset_folder = '..'
results_folder = f'hf_svp_yk_results'
models_folder = f'{results_folder}/optimized_models'

# create results folder if it doesn't already exist
os.makedirs(results_folder, exist_ok=True)
os.makedirs(models_folder, exist_ok=True)

# initialize dataframe to store metrics
R2_df = pd.DataFrame(index=varNames, columns=[])
mae_df = pd.DataFrame(index=varNames, columns=[])

# define problematic molecules to remove from all datasets
prob_mols = [690]

# Get k-fold number from command line argument
k = sys.argv[1]

# =============================================================================
# Plot Configuration
# =============================================================================
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1,1))
plt.rcParams["font.family"] = "Serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
rc('axes', labelsize='16')
rc('xtick', labelsize='14')
rc('ytick', labelsize='14')
rc('legend', fontsize='12')
mpl.rcParams['lines.linewidth'] = 2
plt.rcParams["savefig.pad_inches"]=0.02

# =============================================================================
# Auxiliary Functions
# =============================================================================

def normalize(inputArray,skScaler=None,method='Standardization',reverse=False):
    """
    normalize() normalizes (or unnormalizes) inputArray using the method
    specified and the skScaler provided.

    Parameters
    ----------
    inputArray : np array
        Array to be normalized. If dim>1, array is normalized column-wise.
    skScaler : scikit-learn preprocessing object or None
        Scikit-learn preprocessing object previosly fitted to data. If None,
        the object is fitted to inputArray.
        Default: None
    method : string, optional
        Normalization method to be used.
        Methods available:
            . Standardization - classic standardization, (x-mean(x))/std(x)
            . MinMax - scale to range (0,1)
            . LogStand - standardization on the log of the variable,
                         (log(x)-mean(log(x)))/std(log(x))
            . Log+bStand - standardization on the log of variables that can be
                           zero; uses a small buffer,
                           (log(x+b)-mean(log(x+b)))/std(log(x+b))
        Defalt: 'Standardization'
    reverse : bool
        Whether  to normalize (False) or unnormalize (True) inputArray.
        Defalt: False

    Returns
    -------
    inputArray : np array
        Normalized (or unnormalized) version of inputArray.
    skScaler : scikit-learn preprocessing object
        Scikit-learn preprocessing object fitted to inputArray. It is the same
        as the inputted skScaler, if it was provided.

    """
    # If inputArray is a labels vector of size (N,), reshape to (N,1)
    if inputArray.ndim==1:
        inputArray=inputArray.reshape((-1,1))
        warnings.warn('Input to normalize() was of shape (N,). It was assumed'\
                      +' to be a column array and converted to a (N,1) shape.')
    # If skScaler is None, train for the first time
    if skScaler is None:
        # Check method
        if method=='Standardization' or method=='MinMax': aux=inputArray
        elif method=='LogStand': aux=np.log(inputArray)
        elif method=='Log+bStand': aux=np.log(inputArray+10**-3)
        else: raise ValueError('Could not recognize method in normalize().')
        if method!='MinMax':
            skScaler=preprocessing.StandardScaler().fit(aux)
        else:
            skScaler=preprocessing.MinMaxScaler().fit(aux)
    # Do main operation (normalize or unnormalize)
    if reverse:
        # Rescale the data back to its original distribution
        inputArray=skScaler.inverse_transform(inputArray)
        # Check method
        if method=='LogStand': inputArray=np.exp(inputArray)
        elif method=='Log+bStand': inputArray=np.exp(inputArray)-10**-3
    elif not reverse:
        # Check method
        if method=='Standardization' or method=='MinMax': aux=inputArray
        elif method=='LogStand': aux=np.log(inputArray)
        elif method=='Log+bStand': aux=np.log(inputArray+10**-3)
        else: raise ValueError('Could not recognize method in normalize().')
        inputArray=skScaler.transform(aux)
    # Return
    return inputArray,skScaler

def buildGP(X_Train,Y_Train,gpConfig={}):
    """
    buildGP() builds and fits a GP model using the training data provided.

    Parameters
    ----------
    X_Train : np array (N,K)
        Training features, where N is the number of data points and K is the
        number of independent features (e.g., sigma profile bins).
    Y_Train : np array (N,1)
        Training labels (e.g., property of a given molecule).
    gpConfig : dictionary, optional
        Dictionary containing the configuration of the GP. If a key is not
        present in the dictionary, its default value is used.
        Keys:
            . kernel : string
                Kernel to be used. One of:
                    . 'RBF' - gpflow.kernels.RBF()
                    . 'RQ' - gpflow.kernels.RationalQuadratic()
                    . 'Matern32' - gpflow.kernels.Matern32()
                    . 'Matern52' - gpflow.kernels.Matern52()
                The default is 'RQ'.
            . useWhiteKernel : boolean
                Whether to use a White kernel (gpflow.kernels.White).
                The default is True.
            . trainLikelihood : boolean
                Whether to treat the variance of the likelihood of the modeal
                as a trainable (or fitting) parameter. If False, this value is
                fixed at 10^-5.
                The default is True.
            . alpha : float
                Initial value for the noise variance of the model.
        The default is {}.
    Raises
    ------
    UserWarning
        Warning raised if the optimization (fitting) fails to converge.

    Returns
    -------
    model : gpflow.models.gpr.GPR object
        GP model.

    """
    # Unpack gpConfig
    kernel=gpConfig.get('kernel','RQ')
    useWhiteKernel=gpConfig.get('useWhiteKernel','True')
    trainLikelihood=gpConfig.get('trainLikelihood','True')
    alpha=gpConfig.get('alpha',10**-2)
    # Select and initialize kernel
    if kernel=='RBF':
        gpKernel=gpflow.kernels.SquaredExponential()
    if kernel=='RQ':
        gpKernel=gpflow.kernels.RationalQuadratic()
    if kernel=='Matern32':
        gpKernel=gpflow.kernels.Matern32()
    if kernel=='Matern52':
        gpKernel=gpflow.kernels.Matern52()
    # Add White kernel
    if useWhiteKernel: gpKernel=gpKernel+gpflow.kernels.White()
    # Build GP model    
    model=gpflow.models.GPR((X_Train,Y_Train),gpKernel,noise_variance=alpha)
    # Select whether the likelihood variance is trained
    gpflow.utilities.set_trainable(model.likelihood.variance,trainLikelihood)
    # Build optimizer
    optimizer=gpflow.optimizers.Scipy()
    # Fit GP to training data
    aux=optimizer.minimize(model.training_loss,
                           model.trainable_variables,
                        #    method='L-BFGS-B')
                           method='BFGS')
    # Check convergence
    if aux.success==False:
        warnings.warn('GP optimizer failed to converge.')
    # Output
    return model

def gpPredict(model,X):
    """
    gpPredict() returns the prediction and standard deviation of the GP model
    on the X data provided.

    Parameters
    ----------
    model : gpflow.models.gpr.GPR object
        GP model.
    X : np array (N,K)
        Training features, where N is the number of data points and K is the
        number of independent features (e.g., sigma profile bins).

    Returns
    -------
    Y : np array (N,1)
        GP predictions.
    STD : np array (N,1)
        GP standard deviations.

    """
    # Do GP prediction, obtaining mean and variance
    GP_Mean,GP_Var=model.predict_f(X)
    # Convert to np
    GP_Mean=GP_Mean.numpy()
    GP_Var=GP_Var.numpy()
    # Prepare outputs
    Y=GP_Mean
    STD=np.sqrt(GP_Var)
    # Output
    return Y,STD

def find_zero_cols(property_name, sp_dataset, LOUD=False):
    '''Read the SPs and property data from a specified dataset and return the all-zero columns
    Args:
        property_name: str, name of the property to be predicted
        sp_dataset: str, name of the dataset containing the sigma profiles
    Returns:
        zero_col: np.array, column indices
    '''
    # read generated sigma profiles
    sp_gen = pd.read_csv(f'{dataset_folder}/{sp_dataset}.csv', index_col=0)

    # remove molecules with large sigma ranges
    for prob_idx in prob_mols:
        sp_gen = sp_gen.drop(index=prob_idx)

    # get all valid indices where sp_gen is not nan (x_data)
    valid_x_idx = sp_gen.index[~sp_gen.isna().any(axis=1)]

    # remove columns 1 and 2 and save them to variables
    mol_cas = sp_gen['CAS Number']
    mol_names = sp_gen['Name']
    sp_gen = sp_gen.drop(columns=['CAS Number', 'Name']) # remove columns

    # read target property
    prop = pd.read_csv(f'k-fold-Target-Databases/{property_name}_mlDatabase_Original.csv', index_col=0)
    try:
        prop = prop.drop(columns=['CAS Number', 'Name', 'Temperature /ºC']) # remove columns
    except:
        prop = prop.drop(columns=['CAS Number', 'Name'])
    prop = prop.iloc[:, 0]

    # get common indices between x and y
    y_idx = prop.index
    valid_idx = np.intersect1d(valid_x_idx, y_idx)

    # get x and y data for valid indices
    x_data = sp_gen.loc[valid_idx].to_numpy()
    y_data = prop.loc[valid_idx].to_numpy().reshape(-1, 1)

    # Remove comment columns from SPs, if any
    if type(x_data[1,-1]) == str:
        # remove comment columns
        x_data = x_data[:,:-1].astype(float)

    # find all-zero columns in x-data
    zero_col = np.where(~x_data.any(axis=0))[0]

    # if no all-zero columns are found, return an empy list
    if len(zero_col) == 0:
        return []
        
    # get only zero columns at the edges of the sigma profile
    z_1 = zero_col[0]
    z_2 = zero_col[-1]
    for z in zero_col:
        # compare z with z1
        if np.abs(z-z_1) <= 1:
            z_1 = z
    for z in reversed(zero_col):
        # compare z with z2
        if np.abs(z-z_2) <= 1:
            z_2 = z
    
    # update zero_col to remove columns between z1 and z2
    zero_col = zero_col[np.where((zero_col <= z_1) | (zero_col >= z_2))]

    if LOUD:
        print(f'Number of all-zero columns in {property_name} dataset {sp_dataset}: {len(zero_col)}')

    return zero_col

def extract_data(kfold, property_name, sp_dataset, training=True, return_valid_idx=False, valid_idx=None):
    '''Read the SPs and prpoerty data from a specified dataset and return the training and testing data
    Args:
        kfold: int, kfold number
        property_name: str, name of the property to be predicted
        sp_dataset: str, name of the dataset containing the sigma profiles
        training: bool, if True, read the training property data, else read the testing property data
        return_valid_idx: bool, if True, return the valid indices
        valid_idx: np.array, valid indices
    Returns:
        x_data: np.array, sigma profiles
        y_data: np.array, target property
        valid_idx: np.array, valid indices (optional)
    '''
    # read generated sigma profiles
    sp_gen = pd.read_csv(f'{dataset_folder}/{sp_dataset}.csv', index_col=0)

    # remove molecules with large sigma ranges
    for prob_idx in prob_mols:
        sp_gen = sp_gen.drop(index=prob_idx)

    # get all valid indices where sp_gen is not nan (x_data)
    valid_x_idx = sp_gen.index[~sp_gen.isna().any(axis=1)]

    # remove columns 1 and 2 and save them to variables if any
    try:
        mol_cas = sp_gen['CAS Number']
        mol_names = sp_gen['Name']
        sp_gen = sp_gen.drop(columns=['CAS Number', 'Name']) # remove columns
    except:
        pass

    # read target property
    dataset_name = f'_TrainSet_{kfold}' if training else f'_TestSet_{kfold}'
    prop = pd.read_csv(f'k-fold-Target-Databases/{property_name}_mlDatabase{dataset_name}.csv', index_col=0)
    try:
        prop = prop.drop(columns=['CAS Number', 'Name']) # remove columns
    except:
        pass
    prop = prop.iloc[:, 0]

    # get common indices between x and y
    if valid_idx is None:
        y_idx = prop.index
        valid_idx = np.intersect1d(valid_x_idx, y_idx)
    else:
        valid_idx = valid_idx

    # get x and y data for valid indices
    x_data = sp_gen.loc[valid_idx].to_numpy()
    y_data = prop.loc[valid_idx].to_numpy().reshape(-1, 1)

    # Remove comment columns from SPs, if any
    if type(x_data[1,-1]) == str:
        # remove comment columns
        x_data = x_data[:,:-1].astype(float)

    # get columns that are all zeros
    zero_col = find_zero_cols(property_name, sp_dataset, LOUD=False)
    
    # remove columns that are all zeros from x_data
    x_data = np.delete(x_data, zero_col, axis=1)

    if return_valid_idx == True:
        return x_data, y_data, valid_idx
    else:
        return x_data, y_data
    
# =============================================================================
# Main Script
# =============================================================================

# Iniate timer
ti=time.time()

# loop over properties for selected k-fold
if k not in ['ALL', 'All', 'all']:
    for v, varName in tqdm(enumerate(varNames), desc='\tprop=', total=len(varNames)):
        code = codes[v]
        varUnit = varUnits[v]

        if code in ['S_25', 'VP']:
            labelNorm='LogStand'

        if int(k) == 2: gpConfig['useWhiteKernel']=False 

        print(f'Fitting GP to {varName}')

        # initialize subplot figure for property v
        n_rows = 1;    n_cols = 3
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*3.5, n_rows*3.5))
        axs = axs.ravel()

        # loop over datasets
        for d, dataset in enumerate(datasets):
            ax = axs[d]

            # extract training and testing data
            if d ==0: 
                X_Train, Y_Train, valid_idx_train = extract_data(k, code, dataset, training=True, return_valid_idx=True)
                X_Test, Y_Test, valid_idx_test = extract_data(k, code, dataset, training=False, return_valid_idx=True)
            else:
                # use molecules found in the first dataset to train subsequent datasets
                X_Train, Y_Train = extract_data(k, code, dataset, training=True, valid_idx=valid_idx_train)
                X_Test, Y_Test = extract_data(k, code, dataset, training=False, valid_idx=valid_idx_test)

            # normalize features
            if featureNorm is not None:
                X_Train_N,skScaler_X_Train=normalize(X_Train,method=featureNorm)
                X_Test_N,__=normalize(X_Test,skScaler=skScaler_X_Train,
                                    method=featureNorm)
            else:
                X_Train_N=X_Train
                X_Test_N=X_Test

            # normalize targets
            if labelNorm is not None:
                Y_Train_N,skScaler_Y_Train=normalize(Y_Train,method=labelNorm)
            else:
                Y_Train_N=Y_Train

            # train GP
            model=buildGP(X_Train_N,Y_Train_N,gpConfig=gpConfig)
            
            # Save GP model for no averaging SPs
            if d == 0:
                # Save GP model
                model_file = f"{models_folder}/{code}_model_{k}.pkl"
                with open(model_file, "wb") as f:
                    pickle.dump(model, f)
                # Save skScaler object (will be used to normalize and denormalize for the saved model)
                scaler_file = f"{models_folder}/{code}_scaler_{k}.pkl"
                with open(scaler_file, "wb") as f:
                    if featureNorm is None: skScaler_X_Train=None
                    if labelNorm is None: skScaler_Y_Train=None
                    pickle.dump((skScaler_X_Train, skScaler_Y_Train, featureNorm, labelNorm), f)

            # get GP predictions
            Y_Train_Pred_N,STD_Train=gpPredict(model,X_Train_N)
            Y_Test_Pred_N,STD_Test=gpPredict(model,X_Test_N)
            
            # genormalize
            if labelNorm is not None:
                Y_Train_Pred,__=normalize(Y_Train_Pred_N,skScaler=skScaler_Y_Train,
                                        method=labelNorm,reverse=True)
                Y_Test_Pred,__=normalize(Y_Test_Pred_N,skScaler=skScaler_Y_Train,
                                        method=labelNorm,reverse=True)
            else:
                Y_Train_Pred=Y_Train_Pred_N
                Y_Test_Pred=Y_Test_Pred_N

            # =============================================================================
            # Plots
            # =============================================================================

            # Predictions Scatter Plot
            if code=='VP' or code=='S_25':
                # Get logs of y_data
                log_Y_Train_Pred=np.log(Y_Train_Pred)
                log_Y_Test_Pred=np.log(Y_Test_Pred)
                # Remove Nan values if any
                log_Y_Train=np.log(Y_Train[~np.isnan(log_Y_Train_Pred)])
                log_Y_Test=np.log(Y_Test[~np.isnan(log_Y_Test_Pred)])
                log_Y_Train_Pred=log_Y_Train_Pred[~np.isnan(log_Y_Train_Pred)]
                log_Y_Test_Pred=log_Y_Test_Pred[~np.isnan(log_Y_Test_Pred)]
                # Compute metrics
                R2_Train=metrics.r2_score(log_Y_Train,log_Y_Train_Pred)
                R2_Test=metrics.r2_score(log_Y_Test,log_Y_Test_Pred)
                MAE_Train=metrics.mean_absolute_error(Y_Train,Y_Train_Pred)
                MAE_Test=metrics.mean_absolute_error(Y_Test,Y_Test_Pred)
                # Plot
                # plt.figure(figsize=(2.3,2))
                ax.loglog(Y_Train,Y_Train_Pred,'ow',markersize=3,mec='red')
                ax.loglog(Y_Test,Y_Test_Pred,'^b',markersize=2)
            else:
                # Remove Nan values if any
                Y_Train=Y_Train[~np.isnan(Y_Train_Pred)]
                Y_Test=Y_Test[~np.isnan(Y_Test_Pred)]
                Y_Train_Pred=Y_Train_Pred[~np.isnan(Y_Train_Pred)]
                Y_Test_Pred=Y_Test_Pred[~np.isnan(Y_Test_Pred)]
                # Compute metrics
                R2_Train=metrics.r2_score(Y_Train,Y_Train_Pred)
                R2_Test=metrics.r2_score(Y_Test,Y_Test_Pred)
                MAE_Train=metrics.mean_absolute_error(Y_Train,Y_Train_Pred)
                MAE_Test=metrics.mean_absolute_error(Y_Test,Y_Test_Pred)
                # Plot
                # ax.figure(figsize=(2.3,2))
                ax.plot(Y_Train,Y_Train_Pred,'ow',markersize=3,mec='red')
                ax.plot(Y_Test,Y_Test_Pred,'^b',markersize=2)

            # Plot 1:1 line
            lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
            ax.axline((lims[0], lims[0]), (lims[1], lims[1]), color='k', linestyle='--', linewidth=1)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_xlabel('Exp. ' + varName, weight='bold')
            ax.set_ylabel('Pred. ' + varName, weight='bold')
            ax.set_title(dataset_names[d] + ' SPs', weight='bold')
            ax.set_aspect('equal', adjustable='box')

            ax.text(0.03,0.93,
                    'MAE = '+'{:.2f} '.format(MAE_Train)+varUnit,
                    horizontalalignment='left',
                    transform=ax.transAxes,c='r')
            ax.text(0.03,0.85,
                    'MAE = '+'{:.2f} '.format(MAE_Test)+varUnit,
                    horizontalalignment='left',
                    transform=ax.transAxes,c='b')
            ax.text(0.97,0.11,'$R^2$ = '+'{:.2f}'.format(R2_Train),
                    horizontalalignment='right',
                    transform=ax.transAxes,c='r')
            ax.text(0.97,0.03,'$R^2$ = '+'{:.2f}'.format(R2_Test),
                    horizontalalignment='right',
                    transform=ax.transAxes,c='b')
            
            # save metrics to dataframe
            R2_df.loc[varName, f'{dataset_names[d]}-Train'] = R2_Train
            R2_df.loc[varName, f'{dataset_names[d]}-Test'] = R2_Test
            mae_df.loc[varName, f'{dataset_names[d]}-Train'] = MAE_Train
            mae_df.loc[varName, f'{dataset_names[d]}-Test'] = MAE_Test

            # save metrics to csv
            R2_df.to_csv(f'{results_folder}/R2_arr_{k}.csv')
            mae_df.to_csv(f'{results_folder}/MAE_arr_{k}.csv')

        plt.tight_layout()
        plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.1, wspace=0.4, hspace=0.4)
        fig.savefig(f'{results_folder}/{code}_Pred_{k}.png',bbox_inches='tight')

        # Print elapsed time
        tf=time.time()
        print('Time elapsed: '+'{:.2f}'.format(tf-ti)+' s')

    # save metrics to csv
    R2_df.to_csv(f'{results_folder}/R2_arr_{k}.csv')
    mae_df.to_csv(f'{results_folder}/MAE_arr_{k}.csv')

## Calculate averaged metrics over all k-folds
if k in ['all', 'All', 'ALL']:
    # Initialize dataframe to store averaged metrics
    avg_R2_df = pd.DataFrame(index=varNames, columns=[])
    std_R2_df = pd.DataFrame(index=varNames, columns=[])
    avg_mae_df = pd.DataFrame(index=varNames, columns=[])
    std_mae_df = pd.DataFrame(index=varNames, columns=[])

    # Loop over all properties
    for v, varName in enumerate(varNames):
        code = codes[v]
        varUnit = varUnits[v]
        
        print(f'Getting averaged R2 for {varName}')

        for d, dataset in tqdm(enumerate(datasets), desc='Datasets', total=len(datasets)):
            # initialize empty lists to store R2 and MAE data
            R2s_train = []
            R2s_test = []
            MAEs_train = []
            MAEs_test = []

            for k in tqdm(range(10)):
                # read R2 and MAE data
                R2_df = pd.read_csv(f'{results_folder}/R2_arr_{k}.csv', index_col=0)
                mae_df = pd.read_csv(f'{results_folder}/MAE_arr_{k}.csv', index_col=0)

                # get R2 and MAE data for property v and dataset d
                R2_train = R2_df[f'{dataset_names[d]}-Train'][varName]
                R2_test = R2_df[f'{dataset_names[d]}-Test'][varName]
                MAE_train = mae_df[f'{dataset_names[d]}-Train'][varName]
                MAE_test = mae_df[f'{dataset_names[d]}-Test'][varName]

                # add to lists
                R2s_train.append(R2_train)
                R2s_test.append(R2_test)
                MAEs_train.append(MAE_train)
                MAEs_test.append(MAE_test)
            
            # calculate average R2 and MAE
            R2_avg_train = np.mean(R2s_train)
            R2_avg_test = np.mean(R2s_test)
            MAE_avg_train = np.mean(MAEs_train)
            MAE_avg_test = np.mean(MAEs_test)

            # Calculate standard deviation
            R2_std_train = np.std(R2s_train)
            R2_std_test = np.std(R2s_test)
            MAE_std_train = np.std(MAEs_train)
            MAE_std_test = np.std(MAEs_test)

            # save to dataframe
            avg_R2_df.loc[varName, f'{dataset_names[d]}-Train'] = R2_avg_train
            avg_R2_df.loc[varName, f'{dataset_names[d]}-Test'] = R2_avg_test
            std_R2_df.loc[varName, f'{dataset_names[d]}-Train'] = R2_std_train
            std_R2_df.loc[varName, f'{dataset_names[d]}-Test'] = R2_std_test

            avg_mae_df.loc[varName, f'{dataset_names[d]}-Train'] = MAE_avg_train
            avg_mae_df.loc[varName, f'{dataset_names[d]}-Test'] = MAE_avg_test
            std_mae_df.loc[varName, f'{dataset_names[d]}-Train'] = MAE_std_train
            std_mae_df.loc[varName, f'{dataset_names[d]}-Test'] = MAE_std_test

    # save averaged metrics to csv
    avg_R2_df.to_csv(f'{results_folder}/avg_R2.csv')
    std_R2_df.to_csv(f'{results_folder}/std_R2.csv')
    avg_mae_df.to_csv(f'{results_folder}/avg_mae.csv')
    std_mae_df.to_csv(f'{results_folder}/std_mae.csv')
