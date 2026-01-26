<!--
# Now or never

This repository contains code and data for [Optimal and sub-optimal temporal decisions can explain procrastination in a real-world task](https://escholarship.org/uc/item/2mg517js) accepted at the Cognitive Science Society 2024 meeting. 
-->
Authors: [Sahiti Chebolu](https://sahitic.github.io/) and [Peter Dayan](https://www.mpg.de/12309357/biologische-kybernetik-dayan)

## Abstract
Procrastination is a universal phenomenon, with a significant proportion of the population reporting interference and even harm from such delays. Why do people put off tasks despite what are apparently their best intentions, and why do they deliberately defer in the face of prospective failure? Past research shows that procrastination is a heterogeneous construct with possibly diverse causes. To grapple with the complexity of the topic, we construct a taxonomy of different types of procrastination and potential sources for each type. We simulate completion patterns from three broad model types: exponential or inconsistent temporal discounting, and waiting for interesting tasks; and provide some preliminary evidence, through comparisons with real-world data,  f the plausibility of multiple types of, and pathways for, procrastination.

## Installation

1. clone repository \
   `git clone git@github.com:SahitiC/delay_cogsci2026.git`
2. create and activate python virtual environment using your favourite environment manager (pip, conda etc) \
   for python virtual environments:\
   create: `python3 -m venv .venv`\
   activate: for macOS & Linux, `source .venv/bin/activate` and for Windows, `.venv\Scripts\activate`
   for conda environments:\
   create: `conda create -n venv`\
   activate: `conda activate venv`
3. install packages in requirments.txt: \
   for pip: \
   `pip install -r requirements.txt` \
   for conda: \
   `conda config --add channels conda-forge` \
   `conda install --yes --file requirements.txt`

## Usage

1. First run the data preprocessing script to plot example trajectories from data to reproduce Figure 1A:\
   The code aggregates trajectories over weeks and filters participants. Resulting dataframe is in data_preprocessed.csv
   <code>
   python data_preprocessing.py
   </code>
2. Then run the RL model to generate simulated trajectories and reproduce Figures 1B-D:
   <code>
   python simulations.py
   </code>
3. Fit model to student trajectories and test recovery of fit params:
   <code>
   python model_fitting.py
   python recovery_fits.py
   python recovery_results.py
   </code>
4. Inspect fitted result and do posterior predictive checks, outputs plots in Figures 2 and 3:
   <code>
   python model_fitting_result.py
   </code>
5. Carry out statistical analyses in paper and relate fitted parameters to other measures in data; reproduce Tables 1-4:
   <code>
   python statistics.py
   </code>
        
## Description

1. zhang_ma_data.csv - data from [Zhang and Ma 2024](https://www.nature.com/articles/s41598-024-65110-4). Consists of data from 193 students in a psychology course

2. data_preprocessed.csv - filtered data with delta and cumulative progress over weeks and days

3. data_to_fit_lst.npy - numpy file containing trajectories for fitting model (load with the command `np.load("data_to_fit_lst.npy", allow_pickle=True))

4. fit_individual_mle.npy, fit_params_mle.npy - model fitting results and fitted params

5. recovery_fits_mle.npy - recovery results

6.  plots/ - folder containing all plots as vector images and final figures in the paper

Modules containing some helper functions for further steps: 

7. mdp_algms.py - functions for algorithms that find the optimal policy in MDPs, based on dynamic programming 

8. task_structure.py - functions for constructing reward/ effort functions based on various reward schedules and transition functions based on the transition structure in the different models

9. constants.py - define few shared constants over all the models (states, actions, horizon, effort, shirk reward)

10. gen_data.py -  simulate trajectories from model

11. likelihoods.py - functions for calculating likelihood of trajectories under model

12. regression.py - functions for doing error regression 

13. helper.py - contains some misc. helper functions

The following modules generate the main results in the paper

13. simulations.py - simulate and plot example trajectories while varying params

14. model_fitting.py, model_fitting.sh - fit model to data (contains optio to paralellise), and associated bash script to run on cluster

15. model_fitting_results.py - inspect model fitting results, do posterior predictive checks

16. recovery_fits.py, recovery_fits.sh - run recovery analysis (also paralellised)

17. recovery_results.py - run recovery analysis

18. statistics.py - run correlation and regression analyses 

19. .gitignore - tell git to ignore some local files, please change this based on your local repo

20. requirements.txt - python packages required to run these files

## Citation

<!--
If you found this code or paper helpful, please cite us as:

Chebolu, S., & Dayan, P. (2024). Optimal and sub-optimal temporal decisions can explain procrastination in a real-world task. Proceedings of the Annual Meeting of the Cognitive Science Society, 46. Retrieved from <https://escholarship.org/uc/item/2mg517js> 

<code>
 @article{chebolu2024optimal, 
  title={Optimal and sub-optimal temporal decisions can explain procrastination in a real-world task}, 
  author={Chebolu, Sahiti and Dayan, Peter}, 
  booktitle={Proceedings of the Annual Meeting of the Cognitive Science Society}, 
  volume={46}, 
  year={2024} 
}
</code>
-->

## Contact

For any questions or comments, please contact us at <sahiti.chebolu@tuebingen.mpg.de>
