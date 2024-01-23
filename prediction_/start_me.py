# This function creates the argument list and launches job managers

import numpy
import glob
import re
import argparse
from os import path
import os    # for file operations
import socket   # for netowrk hostname
import numpy
import argparse  # for command-line arguments
import subprocess   # for launching detached processes on a local PC
import sys      # to set exit codes
# Constants
import pandas as pd
from constants import *


# Define arguments
arg_parser = argparse.ArgumentParser(
    description='Job manager. You must choose whether to resume simulations or restart and regenerate the arguments file')
arg_parser.add_argument('-v', '--version', action='version',
                        version='%(prog)s ' + str(version))
mode_group = arg_parser.add_mutually_exclusive_group(required=True)
mode_group.add_argument('--restart', action='store_true')
mode_group.add_argument('--resume', action='store_true')
# arg_parser.add_argument('--ac', action='store', type=str)
arg_parser.add_argument('--hunch_weak', action='store', type=str)

# Identify the system where the code is running
hostname = socket.gethostname()
if hostname.startswith('maestro-submit'):
    script_name = 'sbatch_tars.sh'
    jobs_count = jobs_count_tars
elif hostname == 'patmos':
    script_name = 'sbatch_t_bayes.sh'
    jobs_count = jobs_count_t_bayes
elif hostname == 'onsager-dbc':
    script_name = 'job_manager.py'
    jobs_count = jobs_count_onsager
elif hostname == "thales.dbc.pasteur.fr":
    script_name = 'job_manager.py'
    jobs_count = jobs_count_chloe
else:
    print('Unidentified hostname "' + hostname +
          '". Unable to choose the right code version to launch. Aborting.')
    exit()


# Analyze if need to restart or resume
input_args = arg_parser.parse_args()
bl_restart = input_args.restart
hunch_weak = input_args.hunch_weak
print(type(hunch_weak))
# ac = input_args.ac
ac = 't2'
# If restart is required, regenerate all files
if bl_restart:
    print("Creating arguments list...")

    # Clear the arguments file
    try:
        os.remove(args_file)
    except:
        pass

    # Clean the logs and output folders
    for folder in ([logs_folder]):
        if os.path.isdir(folder):
            print("Cleaning up the folder: '" + folder + "'.")
            cmd = "rm -rfv " + folder
            try:
                os.system(cmd)
            except Exception as e:
                print(e)

        # Recreate the folder
        try:
            os.makedirs(folder)
        except Exception as e:
            print(e)

    # Clean slurm files in the root folder
    cmd = "rm -fv ./slurm-*"
    try:
        os.system(cmd)
    except Exception as e:
        print(e)
    if ac =='t2' or ac == 't0':
        path_ = r'/pasteur/zeus/projets/p02/hecatonchire/tihana/' + ac + '/'
    if ac == 't15' :
        path_ = r'/pasteur/zeus/projets/p02/hecatonchire/screens/' + ac + '/'

    #Liste_set = []
    #date_pre = pd.read_csv('Date_francesca.csv',delimiter = ';', names =  ['line', 1, 2, 3, 4, 5, 6, 7,8,9,10,11,12,13,14,15,16,17,18,19,20])


    date_pre = pd.read_csv('Date_francesca.csv',delimiter = ';', names =  ['line', 1, 2, 3, 4, 5, 6, 7,8,9,10,11,12,13,14,15,16,17,18,19,20])
    #Liste_set = []
    Liste_set = date_pre.index

    Liste_set = [0,1,2,3]
    # GROUP 1 : with hunch_weak [138,139,140,141,142,143,150,151,152]
    # GROUP 2 : with hunch_weak [45,46,47,48,49,50,51,52,53,54,55,56]
    # GROUP 5 : without hunch_weak [0,2,3,4]
    # GROUPE 8 : without hunch_weak [153,154,155,156,157,158]


    print(Liste_set)

    Dossier = Liste_set
    print(len(Dossier))
    id = 0
    with open(args_file, 'w') as file_object:
        for files in Dossier:
            id += 1
            args_string = '-n=%i --id=%i -a=%s -b=%s \n' % (
                files, id, ac, hunch_weak)
            file_object.write(args_string)

    # Create lock file
    with open(args_lock, 'w'):
        pass

    print("Argument list created. Launching sbatch...")

    line_count = id
else:
    print("Resuming simulation with the same arguments file")


# Launch job

if script_name == 'job_manager.py':
    cmd_str = 'python3 %s' % (script_name)
    popens = []
    pids = []
    for j in range(1, jobs_count + 1):
        cur_popen = subprocess.Popen(["python3", script_name])
        popens.append(cur_popen)
        pids.append(cur_popen.pid)
    print("Launched %i local job managers" % (jobs_count))
    print("PIDs: ")
    print(pids)

    # Collect exit codes
    for j in range(jobs_count):
        popens[j].wait()
    print("All job managers finished successfully")

else:
    # -o /dev/null
    cmd_str = 'sbatch --array=1-%i %s' % (jobs_count, script_name)
    os.system(cmd_str)
