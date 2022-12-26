# MiniAOD_photons_to_ML

This code extracts photon candidates from CMS MiniAOD files and is meant to build datasets to train deep-learning models to separate real photons from fakes.

How to run the code: 

### 1) Setup CMSSW environment 
Further information: https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookSetComputerNode

Build a new directory where you want to do your analysis and run the following commands: 

`export VO_CMS_SW_DIR=/cvmfs/cms.cern.ch`

`source $VO_CMS_SW_DIR/cmsset_default.sh`

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


## ToDo 
This code is still under development and contains only a first skeleton at this point. Only runs over one example file. 
- Implement pre-selection cuts on photon candidate shower shapes which are applied before the MVA
- Make the code save the ECAL RecHits, separately for real and fake photons in an appropriate format (e.g. using numpy)  for the ML studies that we want to do
- Validation steps: plotting of shower shapes seperately for realand fakes 
- Add tracks to our calorimeter images 

Further information about how to read MiniAOD files: https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookMiniAOD2017#4_7_MiniAOD_Analysis_Documentati
