"""
What does this code do?

This Python script is the first part of a two-file system designed to process events and store them for future analysis. The primary purpose of this script is to create two-dimensional weights, and it utilises parallelization techniques to significantly improve the performance.

Key Objectives:
1. Data Retrieval: The code's main goal is to access data located at CMS DAS (Data Aggregation System).
2. Preselection: It selects candidates that are relevant to the analysis.
3. Weight Generation: The primary task is to generate two-dimensional weights based on the data and preselected.

The second file loops through the events once more, using the same preselection criteria. However, in this iteration, the selected events are assigned weights, which will be computed within this file.

"""









import multiprocessing
import ROOT
ROOT.gROOT.SetBatch(True)

# load FWLite C++ libraries
ROOT.gSystem.Load("libFWCoreFWLite.so");
ROOT.gSystem.Load("libDataFormatsFWLite.so");
ROOT.FWLiteEnabler.enable()
from DataFormats.FWLite import Handle, Events

import numpy as np
import matplotlib, matplotlib.pyplot as plt
import matplotlib.colors as colors
import mplhep as hep
import copy 
plt.style.use([hep.style.ROOT])
import os 

import mplhep as hep
import hist
import pickle
import time

start = time.time()



def plot_image(image, title, path):
    print("image is being saved")

    cmap = copy.copy(matplotlib.cm.get_cmap("viridis"))
    cmap.set_under('w')

    image[image<1e-6]=1e-6
    im = plt.imshow(image, norm=colors.LogNorm(vmin=0.01, vmax=image.max()), cmap=cmap, interpolation=None)
  
    plt.colorbar(im, label='Energy deposition [GeV]')
    plt.xlabel("iphi")
    plt.ylabel("ieta")
    plt.title(title)
    plt.savefig(path + ".pdf")
    plt.clf()

def hadronic_fake_candidate(photon_candidate, genJets, genParticles):
    """
    This function checks on truth level if the photon candidate stems from a jet. 
    Loops through all genJets and genParticles and checks if 
    the closest object at generator level is a prompt photon, prompt electron, prompt muon or stems from a jet. 
    Returns True / False 
    """

    min_DeltaR = float(99999)

    photon_vector = ROOT.TLorentzVector()
    photon_vector.SetPtEtaPhiE(photon_candidate.pt(), photon_candidate.eta(), photon_candidate.phi(), photon_candidate.energy())

    jet_around_photon = False 
    for genJet in genJets:
        # build four-vector to calculate DeltaR to photon 
        genJet_vector = ROOT.TLorentzVector()
        genJet_vector.SetPtEtaPhiE(genJet.pt(), genJet.eta(), genJet.phi(), genJet.energy())

        DeltaR = photon_vector.DeltaR(genJet_vector)
        # print("\t\t INFO: gen jet eta, phi, delta R ", genJet.eta(), genJet.phi(), DeltaR)
        if DeltaR < 0.3: 
            jet_around_photon = True 

    is_prompt = False
    pdgId = 0 
    for genParticle in genParticles:

        if genParticle.pt() < 5 or abs(genParticle.eta())>1.508: continue # threshold of 1GeV for interesting particles 

        # build four-vector to calculate DeltaR to photon 
        genParticle_vector = ROOT.TLorentzVector()
        genParticle_vector.SetPtEtaPhiE(genParticle.pt(), genParticle.eta(), genParticle.phi(), genParticle.energy())
        
        DeltaR = photon_vector.DeltaR(genParticle_vector)
        if DeltaR < min_DeltaR and DeltaR < 0.3: 
            min_DeltaR = DeltaR
            pdgId = genParticle.pdgId()
            is_prompt = genParticle.isPromptFinalState()

    prompt_electron = True if (abs(pdgId)==11 and is_prompt) else False 
    prompt_photon = True if (pdgId==22 and is_prompt) else False
    prompt_muon = True if (abs(pdgId)==13 and is_prompt) else False 
    
    if jet_around_photon and not (prompt_electron or prompt_photon or prompt_muon):
        return True
    else:
        return False


#this one is updated so not only return energy but also ieta and iphi as well
def select_recHits(recHits, photon_seed, distance=5):
    """
    This function selects ECAL RecHits around the seed of the photon candidate.
    Selects a square of size 2*distance+1
    """

    seed_i_eta = photon_seed.ieta()
    seed_i_phi = photon_seed.iphi()

    rechits_array = []
    index = 0
    
    for recHit in recHits:
        
        # get crystal indices to see if they are close to our photon 
        raw_id = recHit.detid().rawId()
        ID = ROOT.EBDetId(raw_id)
        i_eta = ID.ieta()
        i_phi = ID.iphi()

        if abs(i_phi-seed_i_phi) > distance or abs(i_eta-seed_i_eta) > distance:
            continue
        
        energy = recHit.energy()
        rechits_array.append( [energy, i_eta, i_phi] )


    
    return rechits_array




def main(file_iteration, path = "", distance=5):

    # object collections we want to read:
    # can look into files via: "edmDumpEventContent filepath" to show all available collections

    RecHitHandle, RecHitLabel = Handle("edm::SortedCollection<EcalRecHit,edm::StrictWeakOrdering<EcalRecHit> >"), "reducedEgamma:reducedEBRecHits" 
    photonHandle, photonLabel = Handle("std::vector<pat::Photon>"), "slimmedPhotons"
    pfs, pfLabel = Handle("vector<pat::PackedCandidate> "), "packedPFCandidates"
    genParticlesHandle, genParticlesLabel = Handle("std::vector<reco::GenParticle>"), "prunedGenParticles"
    genJetsHandle, genJetsLabel = Handle("std::vector<reco::GenJet>"), "slimmedGenJets"

    print("INFO: opening file", path.split("/")[-1])
    events = Events(path)


    real_pt=[]
    real_eta=[]
    fake_pt=[]
    fake_eta=[]


    removed=0
    not_removed=0
    for i,event in enumerate(events):        
        if i==0 or i==1000 or i==30000:
            print("\n \t INFO: processing event", i)
            print("number of removed events",removed)
            print("number of not removed",not_removed)

        event.getByLabel(photonLabel, photonHandle)
        event.getByLabel(RecHitLabel, RecHitHandle)
        event.getByLabel(genParticlesLabel, genParticlesHandle)
        event.getByLabel(genJetsLabel, genJetsHandle)
        event.getByLabel(pfLabel, pfs)

        
        pf_candidates=pfs.product()
        genJets = genJetsHandle.product()
        genParticles = genParticlesHandle.product()

        deltaRs_real=[]
        deltaRs_fake=[]
        for photon in photonHandle.product():

            one_object=[]
            is_real = False 
            is_hadronic_fake = False 
            is_the_criteria_passed=False
            
            #There are 2 set of criterias for the photon candidates on the barrel region:
            #1
            if 100>photon.pt() > 32 and photon.r9() > 0.85 and photon.hadTowOverEm()<0.08 and abs(photon.eta())<1.181 and photon.passElectronVeto()==True:
                    
                    is_the_criteria_passed=True
            #2
            if( 100>photon.pt() > 32 and 0.50 < photon.r9() < 0.85 and photon.hadTowOverEm()<0.08 and photon.sigmaEtaEta()<0.015 
            and photon.trackIso()<6 and photon.photonIso()<4 and abs(photon.eta())<1.181 and photon.passElectronVeto()==True):
            
                    is_the_criteria_passed=True
        
           
            if is_the_criteria_passed==True:
                seed_id = photon.superCluster().seed().seed()
                # only use photon candidates in the ECAL barrel (EB) at this point  
                if seed_id.subdetId() != 1: continue 
                
                # photon.genParticle seems to exist only if it is matched to a gen-level photon
                try: 
                    pdgId = photon.genParticle().pdgId()
                    if pdgId == 22: 
                        is_real = True 
                    else:
                        is_real=False

            
                except ReferenceError:
                    is_hadronic_fake = hadronic_fake_candidate(photon, genJets, genParticles)

                if is_real==True:
                    real_pt.append(photon.pt())
                    real_eta.append(photon.eta())
                if is_hadronic_fake==True:
                    fake_pt.append(photon.pt())
                    fake_eta.append(photon.eta())
    file_path="/home/"
    np.savez(file_path+str(file_iteration)+"data.npz",
                            name1=fake_pt,
                            name2=fake_eta,
                            name3=real_pt,
                            name4=real_eta)
                                            








prefix = "root://xrootd-cms.infn.it//"
text_file = "Files_GJet_Pt-40toInf_DoubleEMEnriched_TuneCP5_13p6TeV_pythia8.txt"

def process_file(file_info):
    file_name, iteration = file_info
    print("Processing file:", file_name)
    main(path=prefix + file_name, distance=5, file_iteration=iteration)

if __name__ == '__main__':
    file_list = []
    with open(text_file, "r") as textfile:
        for i, line in enumerate(textfile):
            file_list.append((line.strip(), i))

    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_cores)

    results = pool.map(process_file, file_list)

    pool.close()
    pool.join()



end = time.time()
print("The time passed is:", (end - start)/60," minutes")
