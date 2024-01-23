| Code and analyses by Chloé Barré

## For the *prediction_impTNT* folder

For group 1 and group 4 exclusively,

launch *start_me.py*:
```
python3 ./start_me.py --restart
```

two things to change:

- The list of group numbers in *start_me.py*,
- thresholds:
  - threshold on hunch:
    - `tresh_hunch = 0.4`
  - threshold on static_bend/Head-cast:
    - `static_tresh_one = 1 - 0.3`
    - `static_tresh_two = 0.75`

## For the *prediction_* folder

For group 1 TNT and group 2, group 5 and group 8 exclusively,

launch *start_me.py* with the following argument:
```
   --hunch_weak: True for group 1 and group 2
                 False or nothing for other group.
```
*e.g.*:
```
python3 ./start_me.py --restart --hunch_weak True
```

two things to change:

- The list of group numbers in *start_me.py*,
- thresholds:
  - threshold on hunch:
    - `tresh_hunch = 0.4`
  - threshold on static_bend/Head-cast:
    - `static_tresh_one = 1 - 0.3`
    - `static_tresh_two = 0.75`


## Results

Send the following files to Dylan and Tihana:

* *Larva_behavior*
* *Probability*
* *Proba_data_*
* *Ethogramme*
* all *PREDICTION_* files

## Threshold file

https://docs.google.com/spreadsheets/d/1Ardk-SyFludRvWOo5o3a3uVy0OMOKE11d2ta-h_Gkp0/edit?usp=sharing

## Installation

On Maestro:
```
module load Python/3.10.7
python3 -m pip install numpy pandas h5py "scikit-learn==1.1.2" matplotlib seaborn
```
