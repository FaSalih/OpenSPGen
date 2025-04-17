"""
sort sigma profiles in a given dataset by the range of charge densities and extract information metrics
"""
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
from matplotlib import rc
from matplotlib.font_manager import FontProperties

sys.path.append('Python')
from lib import RDKit_Wrapper as rdw
from lib import spGenerator as spg

#---------------------------------------
# Configure matplotlib
#---------------------------------------
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))

rc('text', usetex=True)
rc('ps', usedistiller='xpdf')
rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
rc('axes', labelsize=14, titleweight='bold', labelweight='bold')
rc('xtick', labelsize=12)
rc('ytick', labelsize=12)
rc('savefig', dpi=300)

# Font properties for bold text
bold_font = FontProperties(weight='bold', size=14)
title_font = FontProperties(weight='bold', size=16)

#---------------------------------------
# Functions
#---------------------------------------
def get_sigma_ranges(file_path, LOUD=True):
    """Read sigma profiles and sort by the range of charge densities.
    Arguments:
        file_path: str, path to the csv file containing the sigma profiles
        LOUD: bool, print the top 10 molecules with the largest charge density ranges
    Returns:
        sigma_vals: np.ndarray, shape=(n_mols, 2), min and max charge densities
        sigma_ranges: np.ndarray, shape=(n_mols,), range of charge densities
        index: np.ndarray, shape=(n_mols,), indices of the molecules
        names: np.ndarray, shape=(n_mols,), names of the molecules
        sps: np.ndarray, shape=(n_mols, n_sigmas), sigma profiles
    """
    # read sp database
    df = pd.read_csv(file_path,  index_col=0)

    # remove molecules/rows with nan values
    df = df.dropna()

    # extract indices and names
    index = df.index.to_numpy()
    try:
        names = df['Name'].values
    except KeyError:
        names = df.index.to_numpy()

    # extract descriptors
    possible_descriptors = ['SMILES', 'InChi', 'InChiKey', 'CAS Number', 'Response']
    # possible_descriptors = ['smiles', 'chembl_id', 'InChi', 'InChiKey', 'CAS Number']
    all_columns = df.columns.to_numpy()
    all_columns = [col.lower() for col in all_columns]
    for desc in possible_descriptors:
        if desc.lower() in all_columns:
            descriptors = df[desc].values
            descriptor_types = desc
            break

    # get number of HBC atoms from descriptors
    num_hbcs = []
    for mol in range(len(descriptors)):
        try:
            num_hbc = get_num_hbc(descriptors[mol], descriptor_type=descriptor_types, get_size=False)
        except:
            num_hbc = np.nan
        num_hbcs.append(num_hbc)
    num_hbcs = np.array(num_hbcs)

    if len(sys.argv) < 3:
        try:
            # remove molecules with large sigma ranges
            prob_mols = [690]
            for prob_idx in prob_mols:
                df = df.drop(index=prob_idx)
            names = np.delete(names, prob_mols)
            index = np.delete(index, prob_mols)
            num_hbcs = np.delete(num_hbcs, prob_mols)
        except:
            print(f"Attempted to remove ${prob_mols} from dataset, but failed.")
    else:
        print('No molecules removed from the dataset, since 2 arguments were passed from the terminal.')

    # drop unnecessary columns
    bad_columns = ['Name', 'Index', 'CAS Number', 'Notes', 'SMILES', 'InChiKey', 'smiles', 'chembl_id', 'Response']
    for col in bad_columns:
        try:
            df = df.drop(columns=col)
        except:
            pass
    
    # place remaining columns in sp matrix
    sigmas = df.columns.to_numpy().astype(float)
    sps = df.values.astype(float)

    # round sigma values to 3rd decimal
    sigmas = np.round(sigmas, 3)

    # loop over molecules and find non-zero range of charge densities
    sigma_vals = []
    sigma_ranges = []
    for sp in sps:
        idx = np.where(sp > 0)[0]
        sigma_vals.append((sigmas[idx[0]], sigmas[idx[-1]]))
        sigma_ranges.append(np.abs(sigmas[idx[-1]] - sigmas[idx[0]]))
    sigma_vals = np.array(sigma_vals)
    sigma_ranges = np.array(sigma_ranges)

    # sort by range of charge densities (largest to smallest)
    sort_idx = np.argsort(sigma_ranges)[::-1]
    sigma_vals = sigma_vals[sort_idx]
    sigma_ranges = sigma_ranges[sort_idx]
    index = index[sort_idx]
    names = names[sort_idx]
    sps = sps[sort_idx]

    if LOUD:
        # print the top 10 molecules
        print('Molecules with the largest charge density ranges:')
        for i in range(10):
            print(f'\t{names[i]}: {sigma_vals[i]}')

    return sigma_vals, sigma_ranges, index, names, sps, sigmas, num_hbcs

def get_sp_gradient(sig,sp):
    """Calculate the average absolute gradient of the SP
    Arguments:
        sig: np.ndarray, shape=(n_sigmas,), sigma values
        sp: np.ndarray, shape=(n_sigmas,), sigma profile
    Returns:
        avg_grad: float, average absolute gradient of the SP
    """
    # get the gradient of the SP
    grad = np.gradient(sp, sig)
    # get the average absolute gradient
    avg_grad = np.mean(np.abs(grad))
    return avg_grad

def get_distance(sig, sp):
    """Calculate the distance between a given SP and a reference "zero-information" SP using the Hellinger distance
    Arguments:
        sig: np.ndarray, shape=(n_sigmas,), sigma values
        sp: np.ndarray, shape=(n_sigmas,), sigma profile
    Returns:
        distance: float, Hellinger distance between the given SP and the reference SP
    """
    # get area under the provided SP curve
    # A_tot = np.trapz(sp, dx=np.round(np.abs(sig[1]-sig[0]), 3))
    A_tot = np.sum(sp)
    # get zero-charge index
    zero_idx = np.where(sig == 0)[0][0]
    # get the reference SP
    sp_ref = np.zeros_like(sp)
    sp_ref[zero_idx] = A_tot
    # remove small negative values from the SP
    if (sp < 0).any():
        # round small negative values to zero
        if (sp[sp < 0] > -1e-6).all():
            sp[sp < 0] = 0
    # get the Hellinger distance
    distance = np.sqrt(np.sum((np.sqrt(sp) - np.sqrt(sp_ref))**2)/2)
    return distance

def remove_outer_zeros(sig, sp):
    '''Remove the outermost zero bins of the sigma profile (get the asymmetric charge density range from the symmetric SP)
    Args:
        sig: np.ndarray, shape=(n_sigmas,), sigma values
        sp: np.ndarray, shape=(n_sigmas,), sigma profile
    Returns:
        sig_min: float, updated minimum sigma value
        sig_max: float, updated maximum sigma value
    '''
    # get min, max and step size
    sig_min = np.round(sig[0], 3)
    sig_max = np.round(sig[-1], 3)
    d_sig = np.abs(np.round((sig[1] - sig[0]), 3))
    # find all zero bins
    zero_bins = np.where(sp == 0)[0] 
    # find the first and last non-zero bins
    if len(zero_bins) > 0:
        # initialize the first and last non-zero bins
        z1 = zero_bins[0]
        z2 = zero_bins[-1]
        # approach from the right to find z1
        for z in zero_bins:
            if np.abs(z-z1) <= 1:
                z1 = z
        # approach from the left to find z2
        for z in reversed(zero_bins):
            if np.abs(z-z2) <= 1:
                z2 = z
        # update zero bins
        zero_bins = zero_bins[np.where((zero_bins <= z1) | (zero_bins >= z2))[0]]
        # remove the zero bins from sigma
        sig_mod = np.array([sig_min + d_sig*i for i in range(len(sp))])
        sig_mod = np.delete(sig_mod, zero_bins)
        # get limits
        sig_min = sig_mod[0]
        sig_max = sig_mod[-1]
    # return sigma limits
    return np.round(sig_min,3), np.round(sig_max,3)

def get_num_hbc(descriptor, descriptor_type='smiles', get_size=False):
    """Get the number of hydrogen-bonding capable atoms in a molecule from a SMILES or CAS descriptor
    Args:
        descriptor: str, SMILES, InChI, InChIKey or CAS number of the molecule
        descriptor_type: str, type of the descriptor
        get_size: bool, whether or not to return molecule size (MW, N_atoms) or not 
    Returns:
        n_hbc: int, number of hydrogen-bonding capable atoms in the molecule
    """
    # get smiles string
    if descriptor_type.lower() != 'smiles': 
        smilesString, warning = spg.crossCheck(descriptor, descriptor_type)
    else:
        smilesString = descriptor
    # create the molecule
    molecule = rdw.getInitialConformer(smilesString)
    # initialize the number of hydrogen-bonding capable atoms
    n_hbc = 0
    # loop over atoms and check if a HB atom
    for i in range(molecule.GetNumAtoms()-1):
        if rdw.isHB(molecule,i) is not None:
            n_hbc += 1
    if get_size:
        # get number of atoms
        n_atoms = molecule.GetNumAtoms()
        mol_mass = molecule.ExactMolWt()

        return n_hbc, (n_atoms, mol_mass)
    else:
        return n_hbc

def get_cumulative_ranges(file_path, LOUD=0):
    """Read sigma profiles and get charge density ranges at 95% and 99% of the area under the curve
    Arguments:
        file_path: str, path to the csv file containing the sigma profiles
        LOUD: int, 
            0: do not print anything
            1: print the top 5 molecules with the largest charge density ranges (w/ 99.9% area)
            2: print the top 5 molecules with the largest charge density ranges (w/ 99% area)
            3: print the top 5 molecules with the largest charge density ranges (w/ 95% area)
    Returns:
        SP_attributes: pd.DataFrame, columns=['Index', 'Name', 'Area_100', 'Charge_100', 'Sigma_Min_100', 'Sigma_Max_100',
                                                'Area_995', 'Charge_995', 'Sigma_Min_995', 'Sigma_Max_995',
                                                'Area_99', 'Charge_99', 'Sigma_Min_99', 'Sigma_Max_99',
                                                'Area_95', 'Charge_95', 'Sigma_Min_95', 'Sigma_Max_95']
    """
    # get 100% area charge ranges
    sig_vals_100, sig_ranges_100, index, names, sps, sigma, num_hbcs = get_sigma_ranges(file_path, LOUD=False)

    # initialize lists to store 95% and 99% metrics
    n_mols = len(names)
    sig_vals_95 = np.zeros((n_mols, 2))
    sig_vals_99 = np.zeros((n_mols, 2))
    sig_vals_995 = np.zeros((n_mols, 2))
    A_95, A_99, A_995, A_100                = 4*[np.zeros(n_mols)]
    q_95, q_99, q_995, q_100                = 4*[np.zeros(n_mols)]
    hd_95, hd_99, hd_995, hd_100            = 4*[np.zeros(n_mols)]
    grad_95, grad_99, grad_995, grad_100    = 4*[np.zeros(n_mols)]

    # get sigma and step size
    sig = np.sort(np.copy(sigma))
    dsig = np.abs(np.round((sig[1] - sig[0]), 3))

    # check if sigma is symmetric and pad with zeros if it is
    if np.abs(sig[0]) != np.abs(sig[-1]):
        # find max absolute sigma value
        sig_max_abs = np.max(np.abs(sig)) + dsig
        # find number of padding zeros on either side
        n_right_pad = int((sig_max_abs - np.abs(sig[-1]))/dsig)
        n_left_pad = int((sig_max_abs - np.abs(sig[0]))/dsig)
        # get sigma values for padding
        sig_right_pad = sig[-1] + dsig * (np.arange(n_right_pad) + 1)
        sig_left_pad = sig[0] - dsig * np.flip(np.arange(n_left_pad) + 1)
        # get SP padding arrays
        sp_right_pad = np.zeros((n_mols, n_right_pad))
        sp_left_pad = np.zeros((n_mols, n_left_pad))
        # pad sigma and sp arrays
        sig = np.concatenate((sig_left_pad, sig, sig_right_pad))
        sps = np.concatenate((sp_left_pad, sps, sp_right_pad), axis=1)
    # round sigma to 3 digits
    sig = np.round(sig,3)

    # loop over sps and get 95% and 99% charge density ranges
    for m in range(n_mols):
        A = sps[m]
        # Get metrics for SPs considering total area under the curve
        A_tot = np.trapz(A, dx=dsig)
        q_tot = np.sum(A*sig)
        hd_tot = get_distance(sig, A)
        grad_tot = get_sp_gradient(sig, A)
        # save to arrays
        A_100[m] = A_tot
        q_100[m] = q_tot
        hd_100[m] = hd_tot
        grad_100[m] = grad_tot
        # initialize charge range and cumulative area
        s_min = 0; s_max = 0
        # loop over charge densities and get 95% and 99% charge density ranges
        for i in range(len(sig)//2):
            # move sigma min and max by one index
            s_min = np.round(s_min - dsig, 3); idx_min = np.where(sig == s_min)[0][0]
            s_max = np.round(s_max + dsig, 3); idx_max = np.where(sig == s_max)[0][0]
            # get cumulative area under the curve and charge in the new sigma limits
            A_cum = np.trapz(A[idx_min:idx_max+1], dx=dsig)
            q_cum = np.sum(A[idx_min:idx_max+1]*sig[idx_min:idx_max+1])
            # check if the cumulative area is greater than 99% of the total area
            if A_cum/A_tot >= 0.995:
                sig_vals_995[m] = remove_outer_zeros(sig[idx_min:idx_max+1], A[idx_min:idx_max+1])
                A_995[m] = A_cum
                q_995[m] = q_cum
                hd_995[m] = get_distance(sig[idx_min:idx_max+1], A[idx_min:idx_max+1])
                grad_995[m] = get_sp_gradient(sig[idx_min:idx_max+1], A[idx_min:idx_max+1])
            elif A_cum/A_tot >= 0.99:
                sig_vals_99[m] = remove_outer_zeros(sig[idx_min:idx_max+1], A[idx_min:idx_max+1])
                A_99[m] = A_cum
                q_99[m] = q_cum
                hd_99[m] = get_distance(sig[idx_min:idx_max+1], A[idx_min:idx_max+1])
                grad_99[m] = get_sp_gradient(sig[idx_min:idx_max+1], A[idx_min:idx_max+1])
                break # move on to the next molecule
            elif A_cum/A_tot >= 0.95:
                sig_vals_95[m] = remove_outer_zeros(sig[idx_min:idx_max+1], A[idx_min:idx_max+1])
                A_95[m] = A_cum
                q_95[m] = q_cum
                hd_95[m] = get_distance(sig[idx_min:idx_max+1], A[idx_min:idx_max+1])
                grad_95[m] = get_sp_gradient(sig[idx_min:idx_max+1], A[idx_min:idx_max+1])
    
    # create dataframe
    SP_attributes = pd.DataFrame({
        'Index': index,
        'Name': names,
        'Num_HBC': num_hbcs,
        'Area_100': A_100,
        'Charge_100': q_100,
        'Sigma_Min_100': sig_vals_100[:,0],
        'Sigma_Max_100': sig_vals_100[:,1],
        'Sigma_Range_100': np.round(sig_vals_100[:,1]-sig_vals_100[:,0], decimals=3),
        'HD_100': hd_100,
        'Grad_100': grad_100,
        'Area_995': A_995, 
        'Charge_995': q_995,
        'Sigma_Min_995': sig_vals_995[:,0],
        'Sigma_Max_995': sig_vals_995[:,1],
        'Sigma_Range_995': np.round(sig_vals_995[:,1]-sig_vals_995[:,0], 3),
        'HD_995': hd_995,
        'Grad_995': grad_995,
        'Area_99': A_99,
        'Charge_99': q_99,
        'Sigma_Min_99': sig_vals_99[:,0],
        'Sigma_Max_99': sig_vals_99[:,1],
        'Sigma_Range_99': np.round(sig_vals_99[:,1]-sig_vals_99[:,0], 3),
        'HD_99': hd_99,
        'Grad_99': grad_99,
        'Area_95': A_95,
        'Charge_95': q_95,
        'Sigma_Min_95': sig_vals_95[:,0],
        'Sigma_Max_95': sig_vals_95[:,1],
        'Sigma_Range_95': np.round(sig_vals_95[:,1]-sig_vals_95[:,0], 3),
        'HD_95': hd_95,
        'Grad_95': grad_95  
    })

    if LOUD == 1:
        # print the top 5 molecules with the largest charge density ranges (w/ 99.9% area)
        print(f'\nMolecules with the largest charge density ranges (w/ 99.5% area):')
        sort_idx = np.argsort(sig_vals_995[:,1] - sig_vals_995[:,0])[::-1]
        names = names[sort_idx]
        indices = index[sort_idx]
        sig_vals_995 = sig_vals_995[sort_idx]
        for i in range(5):
            print(f'\t{indices[i]}: {names[i]}: {sig_vals_995[i]}')
        print(f'\n\tLargest sigma range = {sig_vals_995[0][1]-sig_vals_995[0][0]} $e^-$')

    if LOUD == 2:
        print(f'\nMolecules with the largest charge density ranges (w/ 99% area):')
        sort_idx = np.argsort(sig_vals_99[:,1] - sig_vals_99[:,0])[::-1]
        names = names[sort_idx]
        indices = index[sort_idx]
        sig_vals_99 = sig_vals_99[sort_idx]
        for i in range(5):
            print(f'\t{indices[i]}: {names[i]}: {sig_vals_99[i]}')
        print(f'\n\tLargest sigma range = {sig_vals_99[0][1]-sig_vals_99[0][0]} $e^-$')

    if LOUD == 3:
        # print the top 5 molecules with the largest charge density ranges (w/ 95% area)
        print(f'\nMolecules with the largest charge density ranges (w/ 95% area):')
        sort_idx = np.argsort(sig_vals_95[:,1] - sig_vals_95[:,0])[::-1]
        names = names[sort_idx]
        indices = index[sort_idx]
        sig_vals_95 = sig_vals_95[sort_idx]
        for i in range(5):
            print(f'\t{indices[i]}: {names[i]}: {sig_vals_95[i]}')
        print(f'\n\tLargest sigma range = {sig_vals_95[0][1]-sig_vals_95[0][0]} $e^-$')

    return SP_attributes

#---------------------------------------
# Main
#---------------------------------------
if __name__ == '__main__':
    # make sure the user has provided the dataset
    if len(sys.argv) > 3:
        print('Usage: python filter_sps.py <dataset> \n or:'+
             ' python filter_sps.py <dataset> <dont_remove_prob_mols>')
        sys.exit(1)

    # get the dataset
    dataset_path = sys.argv[1]

    # get cumulative ranges
    SP_attributes = get_cumulative_ranges(dataset_path, LOUD=2)

    # save to csv
    folder_path = '/'.join(dataset_path.split('/')[:-1])
    if "sp.csv" in dataset_path:
        SP_attributes.to_csv(f"{folder_path}/SP_attributes_mul.csv", index=False)
    elif "sp_NoAv.csv" in dataset_path:
        SP_attributes.to_csv(f"{folder_path}/SP_attributes_mul_noav.csv", index=False)
    else:
        SP_attributes.to_csv(f"{folder_path}/SP_attributes.csv", index=False)
