#!/usr/bin/env python3
import MDAnalysis as mda
import numpy as np
import os

print("Running in folder:", os.getcwd())
print("-"*50)


def validate_traj(traj_idx):
    # build file paths
    top = f'{traj_idx}/setup/top.psf'
    elong_file = f'{traj_idx}/ctf_{traj_idx}_elongation.cor'
    eject_file = f'{traj_idx}/ctf_{traj_idx}_ejection.cor'
    ptf_file = f'{traj_idx}/ptf_{traj_idx}.cor'
    ptf_log = f'{traj_idx}/ptf_{traj_idx}.log'

    # check file existence
    missing_files = []
    if not os.path.exists(elong_file):
        missing_files.append("CTF_elongation")
    if not os.path.exists(eject_file):
        missing_files.append("CTF_ejection")
    if not os.path.exists(ptf_file):
        missing_files.append("PTF_cor")
    if not os.path.exists(ptf_log):
        missing_files.append("PTF_log")
    
    # Check if we have at least elongation and ejection files
    if not os.path.exists(elong_file) or not os.path.exists(eject_file):
        print(f"Traj: {traj_idx}: missing files → [{' '.join(missing_files)}]", end='|')
        # if elong/eject cors are not present, check if growth_progress.log is present
        if not os.path.exists(f'{traj_idx}/growth_progress.log'):
            print("CTF is not running, growth_progress.log is not present")
            return
        else:
            with open(f'{traj_idx}/growth_progress.log', 'r') as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1].strip()
                    parts = last_line.split(',')
                    if len(parts) >= 2:
                        growth_site_index = int(parts[0])
                        print(f"CTF is running, growth is at: {growth_site_index}")
                        return

    # elongation
    u = mda.Universe(top, elong_file, format='CRD')
    protein = u.select_atoms('all')
    dist = [np.linalg.norm(protein[i].position - protein[i+1].position) for i in range(protein.n_atoms - 1)]
    elong_min, elong_max, elong_median = np.min(dist), np.max(dist), np.median(dist)

    # ejection
    u = mda.Universe(top, eject_file, format='CRD')
    protein = u.select_atoms('all')
    dist = [np.linalg.norm(protein[i].position - protein[i+1].position) for i in range(protein.n_atoms - 1)]
    eject_min, eject_max, eject_median = np.min(dist), np.max(dist), np.median(dist)

    # PTF (only if files exist)
    ptf_min, ptf_max, ptf_median = None, None, None
    ptf_finished = False
    ptf_step = 0
    
    if os.path.exists(ptf_file):
        u = mda.Universe(top, ptf_file, format='CRD')
        protein = u.select_atoms('all')
        dist = [np.linalg.norm(protein[i].position - protein[i+1].position) for i in range(protein.n_atoms - 1)]
        ptf_min, ptf_max, ptf_median = np.min(dist), np.max(dist), np.median(dist)

    # Check PTF log file for completion (only if file exists)
    if os.path.exists(ptf_log):
        try:
            with open(ptf_log, 'r') as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1].strip()
                    parts = last_line.split()
                    if len(parts) >= 1:
                        ptf_step = int(float(parts[0]))
                        if ptf_step == 100000000:
                            ptf_finished = True
        except (ValueError, IndexError, FileNotFoundError):
            pass

    # validation - check each process individually
    elong_ok = not (elong_min < 3.32 or elong_max > 4.19)
    eject_ok = not (eject_min < 3.32 or eject_max > 4.19)
    ptf_ok = ptf_min is not None and not (ptf_min < 3.32 or ptf_max > 4.19)
    
    # Check if there are any issues
    issues = []
    if not elong_ok:
        issues.append(f"Elongation: {elong_min:.3f} {elong_max:.3f}")
    if not eject_ok:
        issues.append(f"Ejection: {eject_min:.3f} {eject_max:.3f}")
    if ptf_min is not None and not ptf_ok:
        issues.append(f"PTF: {ptf_min:.3f} {ptf_max:.3f}")
    
    if issues:
        print(f'Traj: {traj_idx}: check.', end=' ')
        print(", ".join(issues))
    else:
        print(f'Traj: {traj_idx}: is fine - ', end='')
        status_parts = []
        status_parts.append(f"Ejection: {'✓' if eject_ok else '✗'}")
        status_parts.append(f"Elongation: {'✓' if elong_ok else '✗'}")
        if ptf_min is not None:
            status_parts.append(f"PTF: {'✓' if ptf_ok else '✗'}")
        else:
            status_parts.append("PTF: missing")
        print(", ".join(status_parts), end='')
    
    # PTF completion status
    if ptf_min is not None:
        if ptf_finished:
            print(" - PTF finished")
        else:
            print(f" - PTF not finished (step: {ptf_step})")
    else:
        print(f" - PTF not finished (step: {ptf_step})")
        # print(" - PTF missing")


# 50 trajectories
for i in range(50):
    validate_traj(i)
