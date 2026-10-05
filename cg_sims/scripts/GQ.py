#!/usr/bin/env python3
import requests, logging, os, sys
import time
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import glob
import MDAnalysis as mda
from scipy.spatial.distance import pdist, squareform
# from topoly import lasso_type  # used pip
import itertools
import concurrent.futures
from MDAnalysis.analysis.distances import distance_array
from typing import Dict, List, Tuple, Set, Optional
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, module="MDAnalysis.coordinates.DCD")
warnings.filterwarnings("ignore", message="Reader has no dt information, set to 1.0 ps")

pd.set_option('display.max_rows', 5000)

class Analysis:
    """
    A class to handel the analyssis of a C-alpha CG trajectory. 
    Current analysis available:
    (1) - Fraction of native contacts (Q)
    (2) - Change of entanglement (G)
    """
    #######################################################################################
    def __init__(self, args):
        """
        Initializes the DataAnalysis class with necessary paths and parameters.

        Parameters:
        ("--outpath", type=str, required=True, help="Path to output directory")
        ("--outname", type=str, required=True, help="base name for output files")
        ("--log", type=str, required=True, help="Path to logging file")
        ("--psf", type=str, required=True, help="Path to CA protein structure file")
        ("--aa", type=str, required=True, help="Path to AA protein structure file")
        ("--cor", type=str, required=True, help="Path to CA native coordinates file")
        ("--dcd", type=str, required=True, help="Path to trajectory to analyze")
        ("--sec_elements", type=str, required=True, help="Path to STRIDE secondary structure elements file")
        ("--start", type=int, required=False, default=0, help="First frame to analyze 0 indexed")
        ("--end", type=int, required=False, default=None, help="Last frame to analyze (0-indexed). If None, goes to end of trajectory.")
        ("--stride", type=int, required=False, default=1, help="Frame stride")
        """

        # parse the parameters 
        self.outpath = args.outpath
        logging.info(f'outpath: {self.outpath}')

        self.outname = args.outname
        logging.info(f'outname: {self.outname}')

        self.psf = args.psf
        logging.info(f' psf: {self. psf}')

        self.aa = args.aa
        logging.info(f'aa: {self.aa}')

        self.cor = args.cor
        logging.info(f'cor: {self.cor}')

        self.dcd = args.dcd 
        logging.info(f'dcd: {self.dcd}')

        self.sec_elements = args.sec_elements
        logging.info(f'sec_elements: {self.sec_elements}')

        self.ref_universe = mda.Universe(self.psf, self.cor, format='CRD')
        # print(f'ref_universe: {self.ref_universe}')

        self.aa_universe = mda.Universe(self.aa)
        # print(f'aa_universe: {self.aa_universe}')

        self.traj_universe = mda.Universe(self.psf, self.dcd, format='DCD')
        # print(f'traj_universe: {self.traj_universe}')

        self.scaling_factor = 1.2

        self.start = args.start
        self.stride = args.stride

        # Determined end frame after loading trajectory
        n_frames = len(self.traj_universe.trajectory)
        self.end = args.end if args.end is not None else n_frames
        self.end = min(self.end, n_frames)  # Optional: clamp to max available

        # validate start frame
        if self.start < 0 or self.start >= n_frames:
            raise ValueError(f"Start frame {self.start} is out of range (0 to {n_frames - 1})")

        if self.start >= self.end:
            raise ValueError(f"Start frame ({self.start}) must be less than end frame ({self.end})")

        self.n_threads = args.n_threads
        print(f'START: {self.start} | END: {self.end} | STRIDE: {self.stride}')

        self.topoly = args.topoly
        if self.topoly == 'True':
            self.topoly = True
        else:
            self.topoly = False
        print(f'User topoly {self.topoly}')
    #######################################################################################

    #######################################################################################
    def Q(self, ):
        print(f'Calculating the fraction of native contacts (Q)')
        logging.info(f'Calculating the fraction of native contacts (Q)')
        """
        Calculate the fraction of native contacts in each frame of the DCD where a native contact is defined between secondary structures 
        and for residues atleast that are atleast 3 residues apart. So if i = 1 then j at a minimum can be 5. 
        For a contact to be present the distance between i and j must be less than 8A in the native structure and in a trajectory frame be less than 1.2*native distance.
        """
        # make directory for Q data if it doesnt exist
        self.Qpath = Path(self.outpath) / 'Q'
        self.Qpath.mkdir(parents=True, exist_ok=True)
        print(f'Output directory (Q): {self.Qpath}')

        # # Get the native distance map for the native state cordinates
        # Step 1: Get the native contacts and the reference distance matrix from all atom structure
        print(f'Step 1: Get the native contacts and the reference distance matrix from all atom structure')
        nc_indexs, native_dist_matrix = self.get_native_contacts_from_all_atom_structure()
        NumNativeContacts = len(nc_indexs)
        print(f'Number of Native Contacts (within Secondary Structure Elements) in the Reference AA structure: {NumNativeContacts}')
        logging.info(f'Number of Native Contacts (within Secondary Structure Elements) in the Reference AA structure: {NumNativeContacts}')

        # Step 2: Analyze each frame of the traj_universe and get the distance map
        print(f'Step 2: Analyze each frame of the traj_universe and calc Q')
        # then determine the fraction of native contacts by those distances less than 1.2*native distance
        Qoutput = {'Time(ns)':[], 'Frame':[], 'FrameNumNativeContacts':[], 'Q':[]}
        for ts in self.traj_universe.trajectory[self.start:self.end:self.stride]:
            frame_coor = self.traj_universe.atoms.positions
            frame_distances = np.triu(squareform(pdist(frame_coor)), k=4)
            
            cond = np.triu(np.where((native_dist_matrix > 0) & (frame_distances <= self.scaling_factor * native_dist_matrix), 1, 0), k=4)

            FrameNumNativeContacts = np.sum(cond)
            
            Q = FrameNumNativeContacts/NumNativeContacts
            
            frame_time = f"{ts.time/1000:.3f}"
            Qoutput['Frame'] += [ts.frame]
            Qoutput['FrameNumNativeContacts'] += [FrameNumNativeContacts]
            Qoutput['Q'] += [Q]
            Qoutput['Time(ns)'] += [frame_time]
        
        # Step 3: save Q output 
        print(f'Step 3: save Q output')
        Qoutput = pd.DataFrame(Qoutput)
        Qoutfile = self.Qpath / f'{self.outname}.Q'
        Qoutput.to_csv(Qoutfile, index=False, float_format='%.3f')
        print(f'SAVED: {Qoutfile}')
        logging.info(f'SAVED: {Qoutfile}')
    #######################################################################################  

    #######################################################################################
    def G(self,):
        print(f'Calculating the fraction of native contacts with a change in Gauss Linking Number (G)')
        logging.info(f'Calculating the fraction of native contacts with a change in Gauss Linking Number (G)')
        """
        Calculate the fraction of native contacts that have a change in entanglement status. 
        These native contacts are not restricted to secondary structures.
        """
        # make directory for G data if it doesnt exist
        self.Gpath = Path(self.outpath) / 'G'
        self.Gpath.mkdir(parents=True, exist_ok=True)
        print(f'Output directory (G): {self.Gpath}')

        # Step 1: load the reference structure and topology and get reference state GaussLinking numbers
        print(f'Step 1: load the reference structure and topology and get reference state GaussLinking numbers')
        ref_coor = self.ref_universe.atoms.positions
        #print(f'ref_coor:\n{ref_coor} {ref_coor.shape}')        
        ref_nc_gdict, ref_dist_matrix = self.GaussLink_Ref(ref_coor)
        Nnative = len(ref_nc_gdict)
        print(f'Nnative: {Nnative}')
        logging.info(f'Nnative: {Nnative}')

        # Step 1a: Initiate the output dataframe that will contain all the detailed change in entanglement information. 
        # the native structure data will be populated into Frame -1 and Time(ns) -1 for reference 
        EntOutput = {'Time(ns)':[], 'Frame':[], 'i':[], 'j':[], 'gn':[], 'Gn':[], 'crossingsN':[], 'gc':[], 'Gc':[], 'crossingsC':[], 'NchangeType':[], 'CchangeType':[]}
        nc_list = []
        for nc, ref_info in ref_nc_gdict.items():
            # print(f'REF: {nc} {ref_info}')
            i,j = nc[0], nc[1]
            nc_list += [[i, j]] # update this list for the contact_mask when looping through the trajectory data
            EntOutput['Time(ns)'] += [-1]
            EntOutput['Frame'] += [-1]
            EntOutput['i'] += [i]
            EntOutput['j'] += [j]
            EntOutput['gn'] += [ref_info['gn']]
            EntOutput['Gn'] += [ref_info['Gn']]
            EntOutput['crossingsN'] += [ref_info['crossingsN']]
            EntOutput['gc'] += [ref_info['gc']]
            EntOutput['Gc'] += [ref_info['Gc']]
            EntOutput['crossingsC'] += [ref_info['crossingsC']]
            EntOutput['NchangeType'] += ['None-Ref']
            EntOutput['CchangeType'] += ['None-Ref']
            
      
        # Step 1b: Initiate the output dataframe that will contain the high level stats for changes in entanglement and the G metric used in higherarchical modeling
        Goutput = {'Time(ns)':[], 'Frame':[], 
                    'Nnative':[], 'N-Loss':[], 'N-LossChiral':[], 'N-Gain':[], 'N-GainChiral':[], 'N-PureChiral':[], 
                    'C-Loss':[], 'C-LossChiral':[], 'C-Gain':[], 'C-GainChiral':[], 'C-PureChiral':[], 'FrameNC':[], 'G':[]}

        # Step 2: loop through the frames and calculate the changes in entanglement
        print(f'Step 2: loop through the frames and calculate the changes in entanglement')
        cpu_cores = os.cpu_count()
        n_threads = min(cpu_cores, self.n_threads)
        logging.info(f"Number of available CPU cores: {cpu_cores}")
        print(f"Number of available CPU cores: {cpu_cores}")
        logging.info(f"Using {n_threads} threads for analysis...")
        print(f"Using {n_threads} threads for analysis..")

        # Define the input data for each frame
        frames_data = []
        for ts in self.traj_universe.trajectory[self.start:self.end:self.stride]:
            frame_coor = self.traj_universe.atoms.positions
            frame_data = (
                frame_coor,                # Current frame coordinates
                nc_list,                   # The contact mask (nc_list)
                ref_dist_matrix,           # The reference distance matrix
                ref_nc_gdict,              # The reference nc_gdict
                ts.time/1000,              # Time in ns (dividing by 1000)
                ts.frame,                  # Frame index
                self.GaussLink,            # Function GaussLink
                self.GetLinkChanges,         # Function GetLinkChanges
                Nnative                     # number of native contacts       
            )
            frames_data.append(frame_data)

        # Use ProcessPoolExecutor to parallelize the work
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_threads) as executor:
            # Map the process_frame function to the frames_data
            results = executor.map(process_frame, frames_data)

            #print(f'Number of results: {len(results)}')#
            # Collect the results
            for change_info, count_info in results:
                #print(f'change_info: {change_info}')

                # Adding values of change_info to the master EntOutput dictionary
                for key in EntOutput:
                    EntOutput[key] += change_info[key]

                frame = change_info['Frame'][0]
                frame_time = change_info['Time(ns)'][0]

                # Adding values of count_info to the master Goutput dictionary
                Goutput['Time(ns)'] += [frame_time]
                Goutput['Frame'] += [frame]
                Goutput['Nnative'] += [Nnative]
                for key, value in count_info.items():
                    Goutput[key] += [value]

        # Step 4a: save the dataframe that contains the detailed information about entanglementts
        print(f'Step 4: Save both the detailed entanglment info file .EntInfo and the stats summary file .G')
        EntOutput = pd.DataFrame(EntOutput)
        #print(f'EntOutput:\n{EntOutput.to_string()}')
        # Entoutfile = os.path.join(self.Gpath, f'{self.outname}.EntInfo')
        # EntOutput.to_csv(Entoutfile, index=False, float_format='%.3f')
        Entoutfile = self.Gpath / f'{self.outname}.EntInfo.pkl'
        EntOutput.to_pickle(Entoutfile)
        print(f'SAVED: {Entoutfile}')
        logging.info(f'SAVED: {Entoutfile}')

        # Step 4b: save the dataframe that contains the high level stats information about entanglement changes in each frame
        # should have 1 line per frame of the trajectory
        Goutput = pd.DataFrame(Goutput)
        #Goutput['G'] /= Nnative
        #print(f'Goutput:\n{Goutput.to_string()}')
        Goutfile = self.Gpath / f'{self.outname}.G'
        Goutput.to_csv(Goutfile, index=False, float_format='%.5f')
        print(f'SAVED: {Goutfile}')
        logging.info(f'SAVED: {Goutfile}')
    #######################################################################################

    #######################################################################################
    def GaussLink(self, coor, ref_dist_matrix, contact_mask=[]):
        """
        Function to calculate the Gaussian Linking values for each native contact in a trajectory frame
        Optimized function to calculate the Gaussian Linking values for each native contact in a structure
        """
        Nterm_thresh = 5
        Cterm_thresh = 5

        # make native contact contact map
        dist_matrix = squareform(pdist(coor))
        # Use ref_dist_matrix to determine native contacts: dist < 1.2 * ref_dist_matrix
        native_cmap = np.triu(np.where((ref_dist_matrix > 0) & (dist_matrix <= self.scaling_factor * ref_dist_matrix), 1, 0), k=4)
        nc_indexs = np.stack(np.nonzero(native_cmap)).T

        # If a contact mask list is specified then only grab those contacts in that list to analyze
        # will speed up things when computing over trajectory frames as i only care about native contcats
        if contact_mask:
            #print(f'contact_mask found and will be used')
            masked_nc_indexs = []
            for (i,j) in nc_indexs:
                if [i, j] in contact_mask:
                    masked_nc_indexs += [[i, j]]
            nc_indexs = np.asarray(masked_nc_indexs)

        l = len(dist_matrix)
        range_l = np.arange(0, l - 1)
        range_next_l = np.arange(1, l)

        coor = coor.astype(np.float32)
        R = 0.5 * (coor[range_l] + coor[range_next_l])
        dR = coor[range_next_l] - coor[range_l]

        # Efficient calculation of dR_cross using broadcasting
        dR_cross = np.cross(dR[:, np.newaxis], dR)

        # Efficient Runit matrix calculation using broadcasting
        diff = R[:, np.newaxis] - R
        norms = np.linalg.norm(diff, axis=2, keepdims=True)
        
        # Avoid division by zero by replacing zeros in norms with a small value
        norms = np.where(norms == 0, 1e-8, norms)  # Replace 0 with small value to avoid division by zero
        norms = norms ** 3

        Runit = diff / norms

        # Use broadcasting for dot product calculation
        dot_matrix = np.einsum('ijk,ijk->ij', Runit, dR_cross)

        # Initialize nc_gdict and list to store entangled native contacts
        nc_gdict = {}
        ent_nc = []
        
        for i, j in nc_indexs:
            loop_range = np.arange(i, j)
            nterm_range = np.arange(Nterm_thresh, i - 5)
            cterm_range = np.arange(j + 6, l - (Cterm_thresh + 1))

            gn_pairs_array = np.array(np.meshgrid(nterm_range, loop_range)).T.reshape(-1, 2)
            gc_pairs_array = np.array(np.meshgrid(loop_range, cterm_range)).T.reshape(-1, 2)

            # Calculate gn and gc values
            gn_val = np.sum(dot_matrix[gn_pairs_array[:, 0], gn_pairs_array[:, 1]]) / (4.0 * np.pi) if gn_pairs_array.size else 0
            gc_val = np.sum(dot_matrix[gc_pairs_array[:, 0], gc_pairs_array[:, 1]]) / (4.0 * np.pi) if gc_pairs_array.size else 0

            # Store native contacts that exceed the threshold
            if np.abs(gn_val) >= 0.6 or np.abs(gc_val) >= 0.6:
                ent_nc.append((i, j))
            
            nc_gdict[(i, j)] = {
                'gn': round(gn_val, 5), 'gc': round(gc_val, 5),
                'Gn': custom_round(gn_val), 'Gc': custom_round(gc_val),
                'crossingsN': '', 'crossingsC': ''
            }

        # Get crossings for native contacts
        if self.topoly:
            res = lasso_type(coor.tolist(), loop_indices=ent_nc, more_info=True, min_dist=[10, 6, 5])
            for nc, top_info in res.items():
                nc_gdict[nc]['crossingsN'], nc_gdict[nc]['crossingsC'] = ','.join(top_info['crossingsN']), ','.join(top_info['crossingsC'])

        return nc_gdict
    #######################################################################################
    def GaussLink_Ref(self, coor, contact_mask=[]):
        """
        Function to calculate the Gaussian Linking values for each native contact in a reference structure
        Optimized function to calculate the Gaussian Linking values for each native contact in a structure
        
        coor: the coordinates of the atoms in the structure
        contact_mask: the list of native contacts in the structure
        """
        Nterm_thresh = 5
        Cterm_thresh = 5

        heavy_atoms = self.aa_universe.select_atoms('protein and not name H*')
        resid_to_index, _, n_residues = get_residue_mapping(self.aa_universe)
        aa_distances = distance_array(heavy_atoms.positions, heavy_atoms.positions)
        contact_matrix = np.zeros((n_residues, n_residues))
        native_dist_matrix = np.zeros((n_residues, n_residues))
        cutoff = 4.5
        contact_set = set()
        LOCAL_SEPARATION = 3
        for i, atom1 in enumerate(heavy_atoms):
            for j, atom2 in enumerate(heavy_atoms):
                if (aa_distances[i, j] <= cutoff and abs(atom1.resid - atom2.resid) > LOCAL_SEPARATION):
                    pair = tuple(sorted([atom1.resid, atom2.resid]))
                    contact_set.add(pair)

        for contact in sorted(contact_set, key=lambda x: x[0]):
            contact_matrix[resid_to_index[contact[0]], resid_to_index[contact[1]]] = 1
            contact_matrix[resid_to_index[contact[1]], resid_to_index[contact[0]]] = 1

        contact_matrix = np.triu(contact_matrix, k=4)
        nc_indexs = np.stack(np.nonzero(contact_matrix)).T
        print(f"Number of Native contact (without Secondary Structure Elements consideration) in the Reference AA structure: {len(nc_indexs)}")

        ca_atoms = self.aa_universe.select_atoms('protein and name CA')
        ca_distances = distance_array(ca_atoms.positions, ca_atoms.positions)
        native_dist_matrix[contact_matrix == 1] = ca_distances[contact_matrix == 1]
        
        # make native contact contact map
        dist_matrix = squareform(pdist(coor))
        # native_cmap = np.triu(np.where(dist_matrix <= 8, 1, 0), k=4)
        # nc_indexs = np.stack(np.nonzero(native_cmap)).T

        # If a contact mask list is specified then only grab those contacts in that list to analyze
        # will speed up things when computing over trajectory frames as i only care about native contcats
        if contact_mask:
            #print(f'contact_mask found and will be used')
            masked_nc_indexs = []
            for (i,j) in nc_indexs:
                if [i, j] in contact_mask:
                    masked_nc_indexs += [[i, j]]
            nc_indexs = np.asarray(masked_nc_indexs)

        l = len(dist_matrix)
        range_l = np.arange(0, l - 1)
        range_next_l = np.arange(1, l)

        coor = coor.astype(np.float32)
        R = 0.5 * (coor[range_l] + coor[range_next_l])
        dR = coor[range_next_l] - coor[range_l]

        # Efficient calculation of dR_cross using broadcasting
        dR_cross = np.cross(dR[:, np.newaxis], dR)

        # Efficient Runit matrix calculation using broadcasting
        diff = R[:, np.newaxis] - R
        norms = np.linalg.norm(diff, axis=2, keepdims=True)
        
        # Avoid division by zero by replacing zeros in norms with a small value
        norms = np.where(norms == 0, 1e-8, norms)  # Replace 0 with small value to avoid division by zero
        norms = norms ** 3

        Runit = diff / norms

        # Use broadcasting for dot product calculation
        dot_matrix = np.einsum('ijk,ijk->ij', Runit, dR_cross)

        # Initialize nc_gdict and list to store entangled native contacts
        nc_gdict = {}
        ent_nc = []
        
        for i, j in nc_indexs:
            loop_range = np.arange(i, j)
            nterm_range = np.arange(Nterm_thresh, i - 5)
            cterm_range = np.arange(j + 6, l - (Cterm_thresh + 1))

            gn_pairs_array = np.array(np.meshgrid(nterm_range, loop_range)).T.reshape(-1, 2)
            gc_pairs_array = np.array(np.meshgrid(loop_range, cterm_range)).T.reshape(-1, 2)

            # Calculate gn and gc values
            gn_val = np.sum(dot_matrix[gn_pairs_array[:, 0], gn_pairs_array[:, 1]]) / (4.0 * np.pi) if gn_pairs_array.size else 0
            gc_val = np.sum(dot_matrix[gc_pairs_array[:, 0], gc_pairs_array[:, 1]]) / (4.0 * np.pi) if gc_pairs_array.size else 0

            # Store native contacts that exceed the threshold
            if np.abs(gn_val) >= 0.6 or np.abs(gc_val) >= 0.6:
                ent_nc.append((i, j))
            
            nc_gdict[(i, j)] = {
                'gn': round(gn_val, 5), 'gc': round(gc_val, 5),
                'Gn': custom_round(gn_val), 'Gc': custom_round(gc_val),
                'crossingsN': '', 'crossingsC': ''
            }

        # # Get crossings for native contacts
        # if self.topoly:
        #     res = lasso_type(coor.tolist(), loop_indices=ent_nc, more_info=True, min_dist=[10, 6, 5])
        #     for nc, top_info in res.items():
        #         nc_gdict[nc]['crossingsN'], nc_gdict[nc]['crossingsC'] = ','.join(top_info['crossingsN']), ','.join(top_info['crossingsC'])

        return nc_gdict, native_dist_matrix
    #######################################################################################

    #######################################################################################
    def GetLinkChanges(self, ref_data, frame_data, frame_time, frame, Nnative):
        """
        Calculate the change types in this frame. Each terminal tail (N or C or both) can have one of these change types
        change # any change in the linking number status
        Loss # a loss in the linking number status but no change in chirality (i.e. 2 to 1 | 1 to 0)
        LossChiral # a loss in the linking number status with a change in chirality (i.e. 2 to -1 | -2 to 1)
        Gain # a gain in the linking number status but no change in chirality (i.e. 1 to 2 | 0 to 1)
        GainChiral # a gain in the linking number status with a change in chirality (i.e. 1 to -2 | -1 to 2)
        PureChiral # a pure switch in chirality of the linking number (i.e. -1 to 1 | 2 to -2)
        """
        # this dictionary will track all the detailed information about changes of entanglement observed at each frame
        changes = {'Time(ns)':[], 'Frame':[], 'i':[], 'j':[], 'gn':[], 'Gn':[], 'crossingsN':[], 'gc':[], 'Gc':[], 'crossingsC':[], 'NchangeType':[], 'CchangeType':[]}

        #QC there should be no non-native contacts in the frame_data
        for nc in frame_data:
            if nc not in ref_data:
                raise ValueError(f'There should be no non-native contacts in frame_data but {nc}')

        # This dictionary will track the counts of the changes to output a less detailed text file
        counts = {'N-Loss':0, 'N-LossChiral':0, 'N-Gain':0, 'N-GainChiral':0, 'N-PureChiral':0, 
                    'C-Loss':0, 'C-LossChiral':0, 'C-Gain':0, 'C-GainChiral':0, 'C-PureChiral':0, 'FrameNC':0, 'G':0}

        # loop through frame native contacts and look for changes
        if len(frame_data) != 0:
            for nc, frame_ent_info in frame_data.items():
                counts['FrameNC'] += 1

                ref_Gn = ref_data[nc]['Gn']
                ref_Gc = ref_data[nc]['Gc']
                
                frame_Gn = frame_ent_info['Gn']
                frame_Gc = frame_ent_info['Gc']
                
                # determine N change type
                Nloss = abs(frame_Gn) < abs(ref_Gn)
                Ngain = abs(frame_Gn) > abs(ref_Gn)
                NnoChange = abs(frame_Gn) == abs(ref_Gn)
                NchiralChange = np.sign(frame_Gn) != np.sign(ref_Gn)
                
                if Nloss and not NchiralChange:
                    NchangeType = 'Loss' # Loss of entanglement but no chiral switch
                    counts['N-Loss'] += 1

                elif Nloss and NchiralChange:
                    NchangeType = 'LossChiral' # Loss of entanglement with chiral switch
                    counts['N-LossChiral'] += 1

                elif Ngain and not NchiralChange:
                    NchangeType = 'Gain' # Gain of entanglement but no chiral switch
                    counts['N-Gain'] += 1

                elif Ngain and NchiralChange:
                    NchangeType = 'GainChiral' # Gain of entanglement with chiral switch
                    counts['N-GainChiral'] += 1

                elif NnoChange and NchiralChange:
                    NchangeType = 'PureChiral' # No change in abs(linking value) but a pure chiral switch
                    counts['N-PureChiral'] += 1
                    
                elif NnoChange and not NchiralChange:
                    NchangeType = 'NoChange' # No change in abs(linking value) and no change in chirality

                Closs = abs(frame_Gc) < abs(ref_Gc)
                Cgain = abs(frame_Gc) > abs(ref_Gc)
                CnoChange = abs(frame_Gc) == abs(ref_Gc)
                CchiralChange = np.sign(frame_Gc) != np.sign(ref_Gc)

                if Closs and not CchiralChange:
                    CchangeType = 'Loss' # Loss of entanglement but no chiral switch
                    counts['C-Loss'] += 1

                elif Closs and CchiralChange:
                    CchangeType = 'LossChiral' # Loss of entanglement with chiral switch
                    counts['C-LossChiral'] += 1

                elif Cgain and not CchiralChange:
                    CchangeType = 'Gain' # Gain of entanglement but no chiral switch
                    counts['C-Gain'] += 1

                elif Cgain and CchiralChange:
                    CchangeType = 'GainChiral' # Gain of entanglement with chiral switch
                    counts['C-GainChiral'] += 1

                elif CnoChange and CchiralChange:
                    CchangeType = 'PureChiral' # No change in abs(linking value) but a pure chiral switch
                    counts['C-PureChiral'] += 1

                elif CnoChange and not CchiralChange:
                    CchangeType = 'NoChange' # No change in abs(linking value) and no change in chirality

                # count if any change occured to track G metric
                if NchangeType != 'NoChange' or CchangeType != 'NoChange':
                    counts['G'] += 1

                #print(nc, ref_data[nc], frame_ent_info, NchangeType, CchangeType)
                
                i,j = nc[0], nc[1]
                changes['Time(ns)'] += [frame_time]
                changes['Frame'] += [frame]
                changes['i'] += [i]
                changes['j'] += [j]
                changes['gn'] += [frame_ent_info['gn']]
                changes['Gn'] += [frame_ent_info['Gn']]
                changes['crossingsN'] += [frame_ent_info['crossingsN']]
                changes['gc'] += [frame_ent_info['gc']]
                changes['Gc'] += [frame_ent_info['Gc']]
                changes['crossingsC'] += [frame_ent_info['crossingsC']]
                changes['NchangeType'] += [NchangeType]
                changes['CchangeType'] += [CchangeType]

        # if no native contacts were present make blank row in dataframe
        else:
            changes['Time(ns)'] += [frame_time]
            changes['Frame'] += [frame]
            changes['i'] += [np.nan]
            changes['j'] += [np.nan]
            changes['gn'] += [np.nan]
            changes['Gn'] += [np.nan]
            changes['crossingsN'] += [np.nan]
            changes['gc'] += [np.nan]
            changes['Gc'] += [np.nan]
            changes['crossingsC'] += [np.nan]
            changes['NchangeType'] += [np.nan]
            changes['CchangeType'] += [np.nan]

        counts["G"] /= Nnative
        """
        TODO: Check
        Note on "G" of frame: currently, we separate N-ter and C-ter, hence, each contact is counted twice for N- and C-ter.
        It looks like G = count["G"]/(2*Nnative) 
        """
        logging.info(f'Time(ns): {frame_time} Frame: {frame} with G: {counts}')
        return changes, counts
    #######################################################################################
    def get_native_contacts_from_all_atom_structure(self,):
        """
        Get native contacts and the reference distance matrix from all atom structure.
        """
        ss_residues = load_secondary_structure_ranges(self.sec_elements)

        heavy_atoms = self.aa_universe.select_atoms('protein and not name H*')
        resid_to_index, _, n_residues = get_residue_mapping(self.aa_universe)
        aa_distances = distance_array(heavy_atoms.positions, heavy_atoms.positions)
        contact_matrix = np.zeros((n_residues, n_residues))
        native_dist_matrix = np.zeros((n_residues, n_residues))
        cutoff = 4.5
        contact_set = set()
        LOCAL_SEPARATION = 3
        for i, atom1 in enumerate(heavy_atoms):
            for j, atom2 in enumerate(heavy_atoms):
                # atom.resid is 1-based index, is consistent with the index in secondary structure file.
                if (aa_distances[i, j] <= cutoff and abs(atom1.resid - atom2.resid) > LOCAL_SEPARATION and
                    atom1.resid in ss_residues and atom2.resid in ss_residues):
                    pair = tuple(sorted([atom1.resid, atom2.resid]))
                    contact_set.add(pair)

        for contact in sorted(contact_set, key=lambda x: x[0]):
            contact_matrix[resid_to_index[contact[0]], resid_to_index[contact[1]]] = 1
            contact_matrix[resid_to_index[contact[1]], resid_to_index[contact[0]]] = 1

        contact_matrix = np.triu(contact_matrix, k=4)
        nc_indexs = np.stack(np.nonzero(contact_matrix)).T

        ca_atoms = self.aa_universe.select_atoms('protein and name CA')
        ca_distances = distance_array(ca_atoms.positions, ca_atoms.positions)
        native_dist_matrix[contact_matrix == 1] = ca_distances[contact_matrix == 1]

        return nc_indexs, native_dist_matrix

## Round GaussLink values
def custom_round(number):
    if number >= 0:
        # For positive numbers, round up if fractional part >= 0.6
        return np.ceil(number) if number % 1 >= 0.6 else np.floor(number)
    else:
        # For negative numbers, round down if the absolute fractional part >= 0.6
        # need to take the abs of the number first else the modulus does work right for negative numbers?
        return np.floor(number) if abs(abs(number) % 1) >= 0.6 else np.ceil(number)

def process_frame(frame_data):
    frame_coor, nc_list, ref_dist_matrix, ref_nc_gdict, frame_time, frame, GaussLink, GetLinkChanges, Nnative = frame_data
    # Call GaussLink function
    t1 = time.time()
    frame_nc_gdict = GaussLink(frame_coor, ref_dist_matrix, contact_mask=nc_list)
    #print(f'FRAME: {frame} GaussLink time: {time.time() - t1}')

    # Call GetLinkChanges function
    t1 = time.time()
    change_info, count_info = GetLinkChanges(ref_nc_gdict, frame_nc_gdict, frame_time, frame, Nnative)
    #print(f'FRAME: {frame} GetLinkChanges time: {time.time() - t1}')
    return change_info, count_info

def get_residue_mapping(universe: mda.Universe) -> Tuple[Dict[int, int], Dict[int, str], int]:
    """
    Get residue mapping from universe.
    
    Args:
        universe: MDAnalysis universe object
        
    Returns:
        Tuple of (resid_to_index, index_to_resname, n_residues)
    """
    residues = universe.select_atoms("protein").residues
    n_residues = len(residues)
    resid_to_index = {res.resid: idx for idx, res in enumerate(residues)}
    index_to_resname = {idx: res.resname for idx, res in enumerate(residues)}
    return resid_to_index, index_to_resname, n_residues 

def load_secondary_structure_ranges(filename):
    """
    Load secondary structure ranges from a file.
    Args:
        filename: Path to the file containing the secondary structure ranges
    Returns:
        ss_residues: Set of residue indices that are in secondary structure elements, in the 1-based index
    """
    ss_residues = set()
    with open(filename) as f:
        for line in f:
            if line.strip():
                parts = line.split()
                start = int(parts[1])
                end = int(parts[2])
                ss_residues.update(range(start, end + 1))
    return ss_residues




############## MAIN #################
def main():
    
    script_name = f'GQ'
    parser = argparse.ArgumentParser(description="Process user specified arguments")
    parser.add_argument("-op","--outpath", type=str, required=True, help="Path to output directory")
    parser.add_argument("-on","--outname", type=str, required=True, help="base name for output files")
    parser.add_argument("-psf","--psf", type=str, required=True, help="Path to CA protein structure file")
    parser.add_argument("-cor","--cor", type=str, required=True, help="Path to CA native coordinates file")
    parser.add_argument("-dcd","--dcd", type=str, required=True, help="Path to trajectory to analyze")
    parser.add_argument("-sec","--sec_elements", type=str, required=True, help="Path to STRIDE secondary structure elements file")
    parser.add_argument("-s","--start", type=int, required=False, help="First frame to analyze 0 indexed", default=0)
    parser.add_argument("-e","--end", type=int, required=False, help="Last frame to analyze 0 indexed", default=None)
    parser.add_argument("--stride", type=int, required=False, help="Frame stride", default=1)
    parser.add_argument("-a","--aa", type=str, required=True, help="Path to AA protein structure file")
    parser.add_argument("--topoly", type=str, required=False, help="True = Use Topoly to find crossings. False = Do not find crossings (much faster)", default='False')
    parser.add_argument("-t", "--n_threads", type=int, required=False, help="Number of threads/processes to use", default=1)
    args = parser.parse_args()

    ## make output folder
    outpath = Path(args.outpath)
    outpath.mkdir(parents=True, exist_ok=True)
    print(f'Output directory: {outpath}')

    ## make logging dir
    logs = outpath / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    print(f'Log directory: {logs}')

    # Setup logging configuration
    logfile = logs / f'{args.outname}.log'
    print(f'logfile: {logfile}')
    logging.basicConfig(filename=logfile, level=logging.INFO, format='%(asctime)s %(message)s')
    logging.info(f'{"#"*50}NEW RUN {script_name}{"#"*50}')

    # initialize the simulation object 
    anal = Analysis(args)

    # Get the fraction of native contacts Q
    anal.Q()

    # Get the Change in entanglement parameter G and its associated info
    anal.G()


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    
    print(f'NORMAL TERMINATION: {end_time - start_time}')
    logging.info(f'NORMAL TERMINATION: {end_time - start_time}')