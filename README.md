# MiniAOD_photons_to_ML

This code extracts photon candidates from CMS MiniAOD files and is meant to build datasets to train deep-learning models to separate real photons from fakes.
The files `CRAB_config.py` and `slim_MiniAODs.py` are needed to slim the MiniAOD files and store locally.

The file "1-weigh.py" loops through the files and creates 2 dimensional weights. 
The file "2-datapreperation.py" loops through the files, apply selection criteria and saves the relevant information for the analysis. 














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
