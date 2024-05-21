import sys
sys.path.insert(1,'C:/Users/littl/Box/Research Codes/SPG-main/Python')
from lib import RDKit_Wrapper_13 as rdw
from rdkit import Chem
from lib import VMD_Wrapper as vmd
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
# from tqdm import tqdm 
from itertools import product
import multiprocessing

# Change default plot font to times new roman
plt.rcParams["font.family"] = "Times New Roman"

cutoffs = [20]

for c in range(len(cutoffs)):
    cutoff = cutoffs[c]
    folder = f'Parameter Optimization - All versions\Parameter Optimization - v20.{c} - HighRes'     

    log_path = f"{folder}/log.txt"
    with open(log_path, 'w') as log:
        log.write('\n')
        
    test_mol = pd.read_csv(f'{folder}/Test molecules.csv', header=0)  
        
    num_mol = len(test_mol['Name'])
    n_runs = 5
    HBC_HBC_range = np.logspace(0,3,100)
    HBC_R_range = np.logspace(0,3,100)

    # def calc_loss(indices):
    #     i, j, mol, run = indices
    #     if HBC_HBC_range[i] < HBC_R_range[j]:
    #         val = [np.nan, np.nan]
    #         with open(log_path, 'a') as log:
    #             log.write(f'i={i}, j={j}\n')
    #         return (i, j, mol, run, val[0], val[1])
    #     else:
    #         mol_name=test_mol['Name'][mol] 
    #         smilesString=test_mol['SmilesStrings'][mol] 
    #         # ref_energy=test_mol['Energy'][mol] 
    #         # energy_tol=np.abs(test_mol['Tol'][mol])
                    
    #         repulsion_params = [HBC_HBC_range[i], HBC_R_range[j], cutoff]
            
    #         molecule, energy , sasa = rdw.generateConformer(smilesString,repulsion_params=repulsion_params,xyzPath=None,calc_energy=True)
                
    #         val = [energy, sasa]
    #         with open(log_path, 'a') as log:
    #             log.write(f'i={i}, j={j}\n')
    #         return (i, j, mol, run, val[0], val[1])

    # if __name__ == '__main__':
    #     # create a multiprocessing pool
    #     with multiprocessing.Pool(processes=30) as pool:
            
    #         print('Passed')
            
    #         # use the map method to apply calc_loss to each combination of indices in parallel
    #         results = pool.map( 
    #                             calc_loss, product(
    #                             range(len(HBC_HBC_range)),
    #                             range(len(HBC_R_range)),
    #                             range(num_mol),
    #                             range(n_runs)
    #                                             )
    #                             )

    #     # save the results to a pickle file
    #     fileObj = open(f'{folder}/results.pickle', 'wb')
    #     pickle.dump(results,fileObj)
    #     fileObj.close()

    #     energy_loss = np.zeros((len(HBC_HBC_range),len(HBC_R_range)))
    #     sasa_loss = np.zeros((len(HBC_HBC_range),len(HBC_R_range)))

    #     for i, j, mol, run, energy, sasa in results:
    #         energy_loss[i, j] += energy
    #         sasa_loss[i, j] += sasa

    #     # save the Loss matrix
    #     print(f'energy_loss = \n{energy_loss}')
    #     energy_loss_df = pd.DataFrame(energy_loss, index = HBC_HBC_range, columns = HBC_R_range)
    #     energy_loss_df.to_csv(f'{folder}/Loss (Energy) as a function of HBC repulsion parameters - {n_runs} runs.csv')

    #     # save with the Loss matrix
    #     print(f'sasa_loss = \n{sasa_loss}')
    #     sasa_loss_df = pd.DataFrame(sasa_loss, index = HBC_HBC_range, columns = HBC_R_range)
    #     sasa_loss_df.to_csv(f'{folder}/Loss (SASA) as a function of HBC repulsion parameters - {n_runs} runs.csv')

    # # -----------------------------------------------------------------------------
    # # Post-processing after CRC Paralellized 
    # # -----------------------------------------------------------------------------
    
    # -----------------------------------------------------------------------------
    # Testing different global variables for molecules with < 2 HB-capable group
    # -----------------------------------------------------------------------------
    for loss_type_ind in range(1):
        print("------------------------------------------------------")
        print(f"Cutoff = {cutoff}")
        print(f"Combined loss function {loss_type_ind+2} ...")
        print("Extracting and re-organizing results ...")

        # loop over molecules and categorize them based on number of HB-capable groups
        loss_type = 'Combined- & SASA'
        loss_types = []
        HB_mol = np.zeros(num_mol) # can the molecule form an internal hydrogen bond?
        for m in range(num_mol):
            smiles = test_mol['SmilesStrings'][m] 
            molecule = Chem.MolFromSmiles(smiles)
            molecule = Chem.AddHs(molecule)
            # check for internal HBs
            if rdw.internalHBs(molecule): 
                HB_mol[m] = 1
                loss_types.append('Combined-')
            else:
                HB_mol[m] = 0
                loss_types.append('$SASA$')
        # loss_type = loss_types[2+loss_type_ind]
        extrema_type = "Max"
        parameter_max = 500
        max_param_ind = np.where(HBC_HBC_range > parameter_max)[0][0]
        ext = lambda array: np.nanmax(array[0:max_param_ind, 0:max_param_ind])

        normalize = lambda arr: (arr - np.nanmin(arr))/(np.nanmax(arr) - np.nanmin(arr)) if (np.nanmax(arr) - np.nanmin(arr)) != 0 else np.array([[(np.nan if r > c else 0) for r in range(arr.shape[0])] for c in range(arr.shape[1])])

        # Calculate loss for each molecule
        results_pk = pd.read_pickle(f'{folder}/results.pickle')
        results = pd.DataFrame({
            'i':[results_pk[i][0] for i in range(len(results_pk))], 
            'j':[results_pk[i][1] for i in range(len(results_pk))], 
            'mol':[results_pk[i][2] for i in range(len(results_pk))], 
            'run':[results_pk[i][3] for i in range(len(results_pk))], 
            'ene':[results_pk[i][4] for i in range(len(results_pk))],
            'sas':[results_pk[i][5] for i in range(len(results_pk))],
            })

        labels = test_mol['Name']
        mol_loss1_dict = {}
        mol_loss2_dict = {}
        mol_loss_norm_dict = {}

        for i in range(num_mol):
            # find samples with molecule type 'i'
            mol_results = np.where(results['mol'] == i)[0]
            
            mol_loss_lin1 = results['ene'][mol_results].to_numpy()
            mol_loss1 = np.reshape(mol_loss_lin1, (len(HBC_HBC_range), len(HBC_R_range), n_runs))
            mol_loss1 = np.sum(mol_loss1, axis=2)
            mol_loss1_dict.update({labels[i]:mol_loss1})
            
            mol_loss_lin2 = results['sas'][mol_results].to_numpy()
            mol_loss2 = np.reshape(mol_loss_lin2, (len(HBC_HBC_range), len(HBC_R_range), n_runs))
            mol_loss2 = np.sum(mol_loss2, axis=2)
            mol_loss2_dict.update({labels[i]:mol_loss2})
            
            if loss_types[i] == 'Combined-':
                mol_loss_norm = normalize(mol_loss2) + normalize(mol_loss1) # (SASA + Energy) with the objective to maximize - range is now [0,1]
            else:
                mol_loss_norm = normalize(mol_loss2)  # (SASA) with the objective to maximize - range is now [0,1]
            mol_loss_norm = normalize(mol_loss_norm) # range is now [0,1]
            mol_loss_norm_dict.update({labels[i]:mol_loss_norm})

        #---------------------------------------------------------------------
        # Finding normalized total loss
        print("Calculating and plotting total normalized loss ...")
            
        sig_labels_HB = []     # molecules with significant ranges in energy OR SASA (not just noise)
        sig_type_HB = []       # list for type of singificant loss 
        sig_labels_NHB = []     # molecules with significant ranges in SASA (no hydrogen bonds)
        sig_type_NHB = []       # list for type of singificant loss 
        for m in range(num_mol):
            if loss_types[m] == 'Combined-':
                mol_name = labels[m]
                mol_loss1 = mol_loss1_dict[mol_name]
                mol_loss2 = mol_loss2_dict[mol_name]

                loss1_range = np.nanmax(mol_loss1) - np.nanmin(mol_loss1)
                loss2_range = np.nanmax(mol_loss2) - np.nanmin(mol_loss2)

                sig1 = loss1_range/np.abs(np.nanmin(mol_loss1)) >=  0.05
                sig2 = loss2_range/np.abs(np.nanmin(mol_loss2)) >=  0.01
                if sig1 or sig2:
                    sig_labels_HB.append(mol_name)
                    if sig1 and not(sig2):
                        sig_type_HB.append(1)
                    elif sig2 and not(sig1):
                        sig_type_HB.append(2)
                    elif sig1 and sig2:
                        sig_type_HB.append(3)
            else:
                mol_name = labels[m]
                mol_loss2 = mol_loss2_dict[mol_name]

                loss2_range = np.nanmax(mol_loss2) - np.nanmin(mol_loss2)

                sig2 = loss2_range/np.abs(np.nanmin(mol_loss2)) >=  0.01
                if sig2:
                    sig_labels_NHB.append(mol_name)
                    sig_type_NHB.append(2)
         
        sig_labels_HB = np.array(sig_labels_HB)
        sig_type_HB = np.array(sig_type_HB)
        num_sig_mol_HB = len(sig_labels_HB)

        sig_labels_NHB = np.array(sig_labels_NHB)
        sig_type_NHB = np.array(sig_type_NHB)
        num_sig_mol_NHB = len(sig_labels_NHB)

        norm_loss_all_HB = np.array([mol_loss_norm_dict[sig_labels_HB[i]] for i in range(num_sig_mol_HB)])
        norm_loss_HB = np.sum(norm_loss_all_HB, axis=0)
        
        norm_loss_all_NHB = np.array([mol_loss_norm_dict[sig_labels_NHB[i]] for i in range(num_sig_mol_NHB)])
        norm_loss_NHB = np.sum(norm_loss_all_NHB, axis=0)

        # finding global extreme using sum of normalized losses
        extrema_HB = np.where(norm_loss_HB == ext(norm_loss_HB))
        n_ext_HB = len(extrema_HB[0]) # number of extrema

        extrema_NHB = np.where(norm_loss_NHB == ext(norm_loss_NHB))
        n_ext_NHB = len(extrema_NHB[0]) # number of extrema

        # plt.figure(figsize=(9*0.75,6*0.9),dpi=300)
        # create subplot with 1 row and 2 columns
        fig, axes = plt.subplots(1, 2, figsize=(14*0.75,6*0.9),dpi=300)

        for l in range(2):
            # # finding global extreme using average positions of parameters
            # ext_idx1 = np.mean([np.where(norm_loss_all[i,:,:] == ext(norm_loss_all[i,:,:]))[0][0] for i in range(len(sig_labels))], dtype=int)
            # ext_idx2 = np.mean([np.where(norm_loss_all[i,:,:] == ext(norm_loss_all[i,:,:]))[1][0] for i in range(len(sig_labels))], dtype=int)
            # extrema = np.reshape([ext_idx1, ext_idx2], (2,1))
            # n_ext = 1 # number of extrema

            ax = axes[l]

            norm_loss = [norm_loss_HB, norm_loss_NHB][l]
            extrema = [extrema_HB, extrema_NHB][l]
            n_ext = [n_ext_HB, n_ext_NHB][l]
            sig_labels = [sig_labels_HB, sig_labels_NHB][l]
            sig_type = [sig_type_HB, sig_type_NHB][l]
            loss_type = ['Combined-', '$SASA$'][l]

            ax.contourf(
                HBC_HBC_range,
                HBC_R_range,
                np.transpose(norm_loss), 
                )
            # cbar = ax.colorbar()
            # cbar.ax.set_ylabel('Loss', fontweight='bold', fontsize=14)

            # adding scatter for global optimum
            ax.scatter(
                [HBC_HBC_range[extrema[0][i]] for i in range(n_ext)],
                [HBC_R_range[extrema[1][i]] for i in range(n_ext)],
                label = f"{extrema_type} Loss",
                color = 'black',
                edgecolors='white',
                marker = 'X',
                s = 200,
                )   
            
            # adding scatters for per-molecule optima
            for m in range(num_mol):
                mol_loss = mol_loss_norm_dict[labels[m]]

                mol_loss_type = loss_types[m]
                if mol_loss_type != loss_type:
                    continue

                significant = labels[m] in sig_labels # is this molecule's loss significant to the normalized loss 
                significance = (sig_type[sig_labels==labels[m]][0] if significant else 0)

                mol_extrema = np.where(mol_loss == ext(mol_loss))
                n_mol_ext = len(mol_extrema[0]) #if n_mol_ext < 10 else 10
                if significant:
                    # cross_color = ['red', 'orange', 'purple'][significance-1]
                    cross_color = ['red', 'royalblue'][l]
                    ax.scatter(
                        [HBC_HBC_range[mol_extrema[0][i]] for i in [0]],
                        [HBC_R_range[mol_extrema[1][i]] for i in [0]],
                        label = f"Significant {extrema_type} Loss (per molecule)" if m==5 else "",
                        color = (cross_color if significant else 'white'),
                        edgecolors = ('white' if significant else 'black'),
                        marker = 'X',
                        s = 150,
                        )
                else:
                    ax.scatter(
                    # [HBC_HBC_range[mol_extrema[0][i]] for i in range(n_mol_ext)],
                    # [HBC_R_range[mol_extrema[1][i]] for i in range(n_mol_ext)],
                    [HBC_HBC_range[mol_extrema[0][i]] for i in [0]],
                    [HBC_R_range[mol_extrema[1][i]] for i in [0]],
                    label = f"Insignificant {extrema_type} Loss (per molecule)" if m==1 else "",
                    color = 'white',
                    edgecolors = 'black',
                    marker = 'X',
                    s = 150,
                    )

            ax.set_xscale('log')
            ax.set_yscale('log')

            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)

            ax.set_xlabel('HBD-HBA Relative Force', fontweight='bold', fontsize=16)
            ax.set_ylabel('HBD-R Relative Force', fontweight='bold', fontsize=16)
            ax.set_title(f'Normalized Total {loss_type} - ${extrema_type}(loss) = {round(ext(norm_loss), 2)}$\n Repulsion Cutoff = {cutoffs[c]} Ang', fontweight='bold', fontsize=14)
            ax.legend(fontsize=12)

            norm_loss_df = pd.DataFrame(norm_loss, index = HBC_HBC_range, columns = HBC_R_range)
            norm_loss_df.to_csv(f'{folder}/Normalized loss ({loss_type}) as a function of HBC repulsion parameters - {n_runs} runs.csv')

        plt.tight_layout()
        plt.savefig(f'{folder}/Normalized loss (Multi) as a function of HBC repulsion parameters - {n_runs} runs.png', dpi = 300, bbox_inches='tight')
        # plt.show()       
             
        #---------------------------------------------------------------------
        # Plotting normalized molecule losses
        print("Plotting normalized molecule loss ...")

        num_row = 4
        num_col = 5
        fig, axes = plt.subplots(num_row, num_col, figsize=(2.5*num_col,2*num_row),dpi=300)

        for i in range(num_mol):
            mol_loss_norm = mol_loss_norm_dict[labels[i]]

            loss_type = loss_types[i]
            sig_labels = [sig_labels_NHB, sig_labels_HB][loss_types[i] == 'Combined-']
            sig_type = [sig_type_NHB, sig_type_HB][loss_types[i] == 'Combined-']

            significant = labels[i] in sig_labels
            significance = (sig_type[sig_labels==labels[i]][0] if significant else 0)

            extrema = [extrema_NHB, extrema_HB][loss_type == 'Combined-']
            n_ext = [n_ext_NHB, n_ext_HB][loss_type == 'Combined-']
            
            ax = axes[i//num_col, i%num_col]
            cf = ax.contourf(
                HBC_HBC_range,
                HBC_R_range,
                np.transpose(mol_loss_norm), 
                levels = np.linspace(0, 1, 11),
                vmin = 0.0, vmax = 1.0,
                )
            
            mol_extrema = np.where(mol_loss_norm == ext(mol_loss_norm))
            n_mol_ext = len(mol_extrema[0])

            cross_color = ['red', 'royalblue'][(loss_type == 'Combined-')-1]
            ax.scatter(
                [HBC_HBC_range[mol_extrema[0][i]] for i in range(n_mol_ext)],
                [HBC_R_range[mol_extrema[1][i]] for i in range(n_mol_ext)],
                label = f"{extrema_type} Loss (per molecule)",
                color = (cross_color if significant else 'white'),
                edgecolors = ('white' if significant else 'black'),
                marker = 'X',
                s = 50,
                )
            
            ax.scatter(
                [HBC_HBC_range[extrema[0][i]] for i in range(n_ext)],
                [HBC_R_range[extrema[1][i]] for i in range(n_ext)],
                label = f"{extrema_type} Loss",
                color = ('black' if loss_type == 'Combined-' else 'slategray'),
                edgecolors='white',
                marker = 'X',
                s = 50,
                )
                
            ax.set_title(f'{labels[i]}')
            
            ax.set_xscale('log')
            ax.set_yscale('log')
            
            ax.set_xlabel('HBD-HBA $F_{rel}$')#, fontweight='bold', fontsize=16)
            ax.set_ylabel('HBD-R $F_{rel}$')#, fontweight='bold', fontsize=16)
            
            cbar=fig.colorbar(cf, ax=ax)

        fig.suptitle(f'Normalized Molecule Multi ({n_runs} runs) - Repulsion Cutoff = {cutoffs[c]} Ang', fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{folder}/Molecule loss (Multi) as a function of HBC repulsion parameters (normalized) - {n_runs} runs.png',bbox_inches='tight')
        # plt.show()

        # -----------------------------------------------------------------------------
        # Saving optimal parameters per molecule
        optim_param = np.zeros((num_mol+2, 3))
        for i in range(num_mol):
            mol_loss = mol_loss_norm_dict[labels[i]]
            
            mol_extrema = np.where(mol_loss == ext(mol_loss))
            
            a = HBC_HBC_range[mol_extrema[0][0]]; b = HBC_R_range[mol_extrema[1][0]]
            if loss_types[i] == 'Combined-':
                sig_type = sig_type_HB
                sig_labels = sig_labels_HB
            else:
                sig_type = sig_type_NHB
                sig_labels = sig_labels_NHB
            significance = (sig_type[sig_labels==labels[i]][0] if labels[i] in sig_labels else 0)
            optim_param[i,:] = [a,b,significance] 
            
        optim_param[-2,:] = [HBC_HBC_range[extrema_NHB[0][0]], HBC_R_range[extrema_NHB[1][0]], np.nan]  # adding global minimum
        optim_param[-1,:] = [HBC_HBC_range[extrema_HB[0][0]], HBC_R_range[extrema_HB[1][0]], np.nan]  # adding global minimum

        optim_param_df = pd.DataFrame(optim_param, index = list(labels)+['Global-NHB', 'Global-HB'], columns = ['HBC_HBC', 'HBC_R', 'Significance'])
        optim_param_df.to_csv(f'{folder}/Optimized (Multi) HBC repulsion parameters.csv')
        

