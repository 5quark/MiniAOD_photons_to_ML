# MiniAOD_photons_to_ML

This code extracts photon candidates from CMS MiniAOD files and is meant to build datasets to train deep-learning models to separate real photons from fakes.
The files `CRAB_config.py` and `slim_MiniAODs.py` are needed to slim the MiniAOD files and store locally.

The file "1-weigh.py" loops through the files and creates 2 dimensional weights. 

T loops through the files, apply selection criteria and saves the relevant information for the analysis. 

After creating the two-dimensional weights in the first file, the file "2-datapreperation.py" focuses on applying preselection criteria and saving specific features of candidates that pass the selection process.


Not all candidate features are preserved; only the following are saved:

1.  The ECAL hits of the candidate (individual energy deposition to each of the crystals around the seed).
2.  Photon candidate's transverse momentum (Pt).
3.  Photon candidate's eta and phi values.
4.  A boolean indicating whether the candidate is real photon or not.
5.  The output from a Boosted Decision Tree (BDT).
6.  Some variables that were used for the studies, Charged Hadron, Neutral hadron and photon isolation variables
7.  This file saves certain features of each particle flow object found within a specified cone. The preserved features are, Pt, eta and phi (relative to the photon), as well as dxy, dz, and charge













How to run the code: 

### 1) Setup CMSSW environment 
Further information: https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookSetComputerNode

Build a new directory where you want to do your analysis and run the following commands: 

`source /cvmfs/cms.cern.ch/cmsset_default.sh`

`cmsrel CMSSW_12_6_0`

`cd CMSSW_12_6_0/src`

`cmsenv`

### 2) Clone repository and run the code
`git clone https://github.com/FloMau/MiniAOD_photons_to_ML.git` (or better: fork it and clone your fork!)

`cd MiniAOD_photons_for_ML`

Now you are ready to go! You can run the script using

`python3 MiniAOD_analyser.py`

After first time with installing, you can skip some of the steps above, just do:

`source /cvmfs/cms.cern.ch/cmsset_default.sh`

`cd CMSSW_12_6_0/src` 

`cmsenv`

`cd MiniAOD_photons_for_ML`

`python3 MiniAOD_analyser.py`
