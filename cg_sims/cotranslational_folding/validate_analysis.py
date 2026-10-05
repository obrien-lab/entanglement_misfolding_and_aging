#!/usr/bin/env python3
import os
import pandas as pd

print("Running in folder:", os.getcwd())
print("-"*50)


def validate_gq_analysis(traj_idx):
    # build file paths
    g_file = f'{traj_idx}/GQ/G/ptf_{traj_idx}.G'
    q_file = f'{traj_idx}/GQ/Q/ptf_{traj_idx}.Q'
    eject_g_file = f'{traj_idx}/GQ/G/ctf_ejection_{traj_idx}.G'
    eject_q_file = f'{traj_idx}/GQ/Q/ctf_ejection_{traj_idx}.Q'
    
    # check file existence
    missing_files = []
    if not os.path.exists(g_file):
        missing_files.append("PTF_G")
    if not os.path.exists(q_file):
        missing_files.append("PTF_Q")
    if not os.path.exists(eject_g_file):
        missing_files.append("eject_G")
    if not os.path.exists(eject_q_file):
        missing_files.append("eject_Q")
    
    if missing_files:
        print(f"Traj: {traj_idx}: missing GQ files → {' '.join(missing_files)}")
        return
    
    # Check G file
    g_valid = False
    g_last_step = None
    try:
        with open(g_file, 'r') as f:
            lines = f.readlines()
            if lines:
                last_line = lines[-1].strip()
                parts = last_line.split(',')
                if len(parts) >= 2:
                    g_last_step = int(parts[1])
                    if g_last_step == 19999:
                        g_valid = True
    except (ValueError, IndexError, FileNotFoundError):
        pass
    
    # Check Q file
    q_valid = False
    q_last_step = None
    try:
        with open(q_file, 'r') as f:
            lines = f.readlines()
            if lines:
                last_line = lines[-1].strip()
                parts = last_line.split(',')
                if len(parts) >= 2:
                    q_last_step = int(parts[1])
                    if q_last_step == 19999:
                        q_valid = True
    except (ValueError, IndexError, FileNotFoundError):
        pass
    
    # Check ejection G file (only existence and non-empty)
    eject_g_valid = False
    try:
        with open(eject_g_file, 'r') as f:
            content = f.read().strip()
            if content:  # file is not empty
                eject_g_valid = True
    except (FileNotFoundError, IOError):
        pass
    
    # Check ejection Q file (only existence and non-empty)
    eject_q_valid = False
    try:
        with open(eject_q_file, 'r') as f:
            content = f.read().strip()
            if content:  # file is not empty
                eject_q_valid = True
    except (FileNotFoundError, IOError):
        pass
    
    # Report results
    if g_valid and q_valid and eject_g_valid and eject_q_valid:
        print(f'Traj: {traj_idx}: GQ analysis complete - PTF_G: ✓, PTF_Q: ✓, Eject_G: ✓, Eject_Q: ✓')
    else:
        print(f'Traj: {traj_idx}: GQ analysis incomplete - ', end='')
        status_parts = []
        if g_valid:
            status_parts.append("PTF_G: ✓")
        else:
            status_parts.append(f"PTF_G: ✗ (step: {g_last_step})")
        
        if q_valid:
            status_parts.append("PTF_Q: ✓")
        else:
            status_parts.append(f"PTF_Q: ✗ (step: {q_last_step})")
        
        if eject_g_valid:
            status_parts.append("Eject_G: ✓")
        else:
            status_parts.append("Eject_G: ✗ (empty/missing)")
        
        if eject_q_valid:
            status_parts.append("Eject_Q: ✓")
        else:
            status_parts.append("Eject_Q: ✗ (empty/missing)")
        
        print(", ".join(status_parts))


# 50 trajectories
for i in range(50):
    validate_gq_analysis(i)
