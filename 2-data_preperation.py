

"""

This is the second file in a two-file system designed for data preparation. After creating the two-dimensional weights in the first file, 
this second file focuses on applying preselection criteria and saving specific features of candidates that pass the selection process.


Not all candidate features are preserved; only the following are retained:

1.  The ECAL hits of the candidate (individual energy deposition to each of the crystals around the seed).
2.  Photon candidate's transverse momentum (Pt).
3.  Photon candidate's eta and phi values.
4.  A boolean indicating whether the candidate is real photon or not.
5.  The output from a Boosted Decision Tree (BDT).
6.  Some variables that were used for the studies, Charged Hadron, Neutral hadron and photon isolation variables
7.  This file saves certain features of each particle flow object found within a specified cone. The preserved features are, Pt, eta and phi (relative to the photon), as well as dxy, dz, and charge

"""











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

    # print("\t\t Photon gen particle PDG ID:", photon_candidate.genParticle().pdgId())

    # this jet loop might be not needed... check later, but doesn't harm at this point 
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
       # print("dir genparticle",dir(genParticle))
        if genParticle.pt() < 5 or abs(genParticle.eta())>1.508: continue # threshold of 1GeV for interesting particles 

        # build four-vector to calculate DeltaR to photon 
        genParticle_vector = ROOT.TLorentzVector()
        genParticle_vector.SetPtEtaPhiE(genParticle.pt(), genParticle.eta(), genParticle.phi(), genParticle.energy())
        
        DeltaR = photon_vector.DeltaR(genParticle_vector)
        # print("\t\t INFO: gen particle eta, phi, delta R ", genParticle.eta(), genParticle.phi(), DeltaR)
        if DeltaR < min_DeltaR and DeltaR < 0.3: 
            min_DeltaR = DeltaR
            pdgId = genParticle.pdgId()

            is_prompt = genParticle.isPromptFinalState()
        # print("\t\t INFO: PDG ID:", pdgId)

    prompt_electron = True if (abs(pdgId)==11 and is_prompt) else False 
    prompt_photon = True if (pdgId==22 and is_prompt) else False
    prompt_muon = True if (abs(pdgId)==13 and is_prompt) else False 
#    if prompt_electron==True:
#        print("prompt electron",prompt_electron)
    if jet_around_photon and not (prompt_electron or prompt_photon or prompt_muon):
        return True
    else:
        return False


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
        #print("Types energy, i_eta, i_phi",type(energy),type(i_eta),type(i_phi))
        rechits_array.append( [float(energy), float(i_eta), float(i_phi)] )


    
    return rechits_array






#A[0]  1 photon candidate and everything associated to this including the pf candidates.  
#A[0][0] Photon ecal hits, pt, eta and phi 
#A[0][0][0] 121 element ecal image
#A[0][0][1] pt
#A[0][0][2] eta
#A[0][0][3] phi
#A[0][0][3] isReal
# 0<i
#A[0][i] Pf objects within the cone 
#A[0][i][0] Pt
#A[0][i][0] eta
#A[0][i][0] phi
#                          rec hits           pt,eta,phi,Weight isReal   , BDT ,   C.,     N. Hadron,  photon iso       pf1               pf2              pf3
#A=[       [     [        [1,2,3,4...     ],     1,2,3,    4 ,   True    , {-1,1},  7     ,8                ,9   ] ]    , [1,2,3,4,5,6] , [1,2,3,4,5,6], [1,2,3,4,5,6]     ] ,     [more candidates similar to first one]   ]
#           ------------------1 photon candidate and everything associated to that------------------------------
#pf object: [pt, eta, phi, dxy, dz, charge,weight]







import os

file_path = "/home/home1/institut_3a/coban/CMSdata/CMSSW_12_6_0/src/MiniAOD_photons_to_ML/copy_paste/1_file/1_updated_e_veto/"
file_iteration = 1

# Initialize empty arrays for concatenation
all_fake_pt = np.array([])
all_fake_eta = np.array([])
all_real_pt = np.array([])
all_real_eta = np.array([])


count=0
for file_name in os.listdir(file_path):
    if file_name.endswith(".npz"):
        file_path_name = os.path.join(file_path, file_name)
        data = np.load(file_path_name)
        # Extract arrays from the loaded file
        fake_pt = data['name1']
        fake_eta = data['name2']
        real_pt = data['name3']
        real_eta = data['name4']

        # Concatenate the arrays
        all_fake_pt = np.concatenate((all_fake_pt, fake_pt))
        all_fake_eta = np.concatenate((all_fake_eta, fake_eta))
        all_real_pt = np.concatenate((all_real_pt, real_pt))
        all_real_eta = np.concatenate((all_real_eta, real_eta))
        count=count+1

print(count)

print("real_pt",len(all_real_pt))
print("real_pt",all_real_pt.shape)
print("real_pt",np.amin(all_real_pt))
print("real_pt",np.amax(all_real_pt))
print("real_pt",np.amin(abs(all_real_eta)))
print("real_pt",np.amax(all_real_eta))

print("real_pt",len(all_real_eta))


counts_fake, xedges_fake, yedges_fake, im_fake = plt.hist2d(all_fake_pt, np.absolute(all_fake_eta), bins=[14, 4], range=[[30, 100], [0, 1.184]])    #,density=True
counts_real, xedges_real, yedges_real, im_real = plt.hist2d(all_real_pt, np.absolute(all_real_eta) , bins=[14, 4], range=[[30, 100], [0, 1.184]])

rows=14
cols=4
weights = np.zeros((rows, cols))
#do something about the cases where fakes are zero. In that case weight would not increase the fakes to the whatever the real value we had. 
fakes_zero=[]
for i in range(14):
    for j in range(4):
        if counts_fake[i,j] !=0:
            weight= counts_real[i,j]/counts_fake[i,j]
            weights[i][j]=weight
        else:
            weights[i][j]=0  
            fakes_zero.append((i,j))

print("weights are: *******************************************",weights)
print("fakes",counts_fake)
print("reals",counts_real)

print("fakeszero*************************** ", fakes_zero)




#The following part is for creating the weights in a normalised manner. I think this makes less sense but I will save them anyway









def main(file_iteration,path = "", distance=5):
     
    time1=time.time()
    #The lists that I will combine and save in the end:
    photon_real=[]
    photon_fake=[]
    x_real=[]
    y_real=[]
    w_real=[]
    pt_real=[]
    bdt_real=[]
    C_hadron_iso_real=[]
    N_hadron_iso_real=[]
    photon_iso_real=[]
    pf_lists_real=[]

    x_fake=[]
    y_fake=[]
    w_fake=[]
    pt_fake=[]
    bdt_fake=[]
    C_hadron_iso_fake=[]
    N_hadron_iso_fake=[]
    photon_iso_fake=[]
    pf_lists_fake=[]

    list_of_lists=[]
    ######################End of the lists that will be saved.

    # object collections we want to read:
    # can look into files via: "edmDumpEventContent filepath" to show all available collections
    RecHitHandle, RecHitLabel = Handle("edm::SortedCollection<EcalRecHit,edm::StrictWeakOrdering<EcalRecHit> >"), "reducedEgamma:reducedEBRecHits" 
    photonHandle, photonLabel = Handle("std::vector<pat::Photon>"), "slimmedPhotons"
    pfs, pfLabel = Handle("vector<pat::PackedCandidate> "), "packedPFCandidates"
    genParticlesHandle, genParticlesLabel = Handle("std::vector<reco::GenParticle>"), "prunedGenParticles"
    genJetsHandle, genJetsLabel = Handle("std::vector<reco::GenJet>"), "slimmedGenJets"
    print("***************************")
    PackedGenParticleHandle, PackedGenParticleLabel = Handle("vector<pat::PackedGenParticle>"), "packedGenParticles" 

    print("**************")
    print("INFO: opening file", path.split("/")[-1])
    events = Events(path)





    for i,event in enumerate(events):
        #"edmDumpEventContent filepath" try this later on to see which ones are defined and also you need to define them like the ones above at lines 130-140
        
        if i==0 or i==1000:
            print("\n \t INFO: processing event", i)

        event.getByLabel(photonLabel, photonHandle)
        event.getByLabel(RecHitLabel, RecHitHandle)
        event.getByLabel(genParticlesLabel, genParticlesHandle)
        event.getByLabel(genJetsLabel, genJetsHandle)
        event.getByLabel(pfLabel, pfs)
        event.getByLabel(PackedGenParticleLabel, PackedGenParticleHandle)

        pf_candidates=pfs.product()
        genJets = genJetsHandle.product()
        genParticles = genParticlesHandle.product()
        """
        for PackedGenParticle in PackedGenParticleHandle.product():
            print("dir(PackedGenParticle)",dir(PackedGenParticle))
            break
        """

      
        for p in genJetsHandle.product() :
            print("p.mother*******************************************",p.mother(0).pdgid())

        deltaRs_real=[]
        deltaRs_fake=[]
        for photon in photonHandle.product():

            one_object=[]
            is_real = False 
            is_hadronic_fake = False 
            is_the_criteria_passed=False
            
            #There are 2 set of criterias for the photon candidates on the barrel region:
            #1                           we don't want track: eveto or something  passeveto==true              
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
                        print("**********************************pdgid:",pdgId)

            
                except ReferenceError:
                    #print("reference error executed*****************************************")
                    is_hadronic_fake = hadronic_fake_candidate(photon, genJets, genParticles)
              #     "should be true otherwise do not save it: hadronic_fake_candidate==True"
#check if this returns true for fakes.
                if is_real == True or is_hadronic_fake==True:
                    flag=False
                    """
                    for i,j in fakes_zero:
                        if i*5 <=photon.pt()<= (i+1)*5 and j*0.42 <=photon.eta()<=(j+1)*0.42:
                            flag=True
                            break
                    """
                    element_i=int(photon.pt()/5)
                    element_j=int(photon.eta()/0.296)
                    if element_i<14 and element_j<4:
                        w=weights[element_i][element_j]
                        flag=False
                    if element_i>=14 or element_j>=4:
                        flag=True
                
                    if flag== False:
                        w=weights[element_i][element_j]
                        # get crystal indices of photon candidate seed:
                        seed_id = ROOT.EBDetId(seed_id)
                        recHits = RecHitHandle.product()
                        rechits_array = select_recHits(photon_seed=seed_id, recHits=recHits, distance=distance)
                        one_object.append(rechits_array)
                        one_object.append(photon.pt())
                        one_object.append(photon.eta())
                        one_object.append(photon.phi())
                        one_object.append(w)
                        one_object.append(is_real)
                        one_object.append(photon.userFloat("PhotonMVAEstimatorRunIIFall17v2Values"))   #bdt values
                        one_object.append(photon.chargedHadronIso())
                        one_object.append(photon.neutralHadronIso())
                        one_object.append(photon.photonIso())
                        #print("*******************************",dir(photon.mother(0)))
                        #Loop through all the PF particles and if they fit in the criteria save their momentums as a multidimensional array.
                        for pf in pf_candidates:
                            if abs(pf.eta()-photon.eta())<0.3:
                                #calculate this by hand and try the speed. 
                                DeltaR = ((pf.eta()-photon.eta())**2+(pf.phi()-photon.phi())**2)**(1/2)
                                if DeltaR < 0.3: 
                                    one_object.append([pf.pt(), photon.eta()-pf.eta(),photon.phi()- pf.phi(),pf.dxy(),pf.dz(),pf.charge()])
                        #save this objet into the list 
                        list_of_lists.append(one_object)   
                              
    time2=time.time()
    print("time passed for one file is:",time2-time1)

    #save it here:   

    for photon in list_of_lists:
        if photon[5]==True:
            photon_real.append(photon)
        if photon[5]==False:
            photon_fake.append(photon)
    #there are more real compared to fakes so let's remove some of them
    photon_real[:len(photon_fake)]
    #now seperate them into real x and fake x etc. 
    for photon in photon_real:
        x_real.append(photon[0])
        y_real.append(photon[5])
        w_real.append(photon[4])
        pt_real.append(photon[1])
        bdt_real.append(photon[6])
        C_hadron_iso_real.append(photon[7]) 
        N_hadron_iso_real.append(photon[8])
        photon_iso_real.append(photon[9])
        pf_lists_real.append([photon[10:]])

    for photon in photon_fake:
        x_fake.append(photon[0])
        y_fake.append(photon[5])
        w_fake.append(photon[4])
        pt_fake.append(photon[1])  
        bdt_fake.append(photon[6]) 
        C_hadron_iso_fake.append(photon[7])
        N_hadron_iso_fake.append(photon[8])
        photon_iso_fake.append(photon[9])
        pf_lists_fake.append([photon[10:]])
    print("*****************SAVING**************************************************************************************************")  
    path="/home/home1/institut_3a/coban/CMSdata/CMSSW_12_6_0/src/MiniAOD_photons_to_ML/copy_paste/1_file/2_updated_e_veto/"  
    file_name=path+str(file_iteration)+".npz"
    """
    np.savez(file_name,
            x_real=x_real,
            y_real=y_real,
            w_real=w_real,
            pt_real=pt_real,
            bdt_real=bdt_real,
            C_hadron_iso_real=C_hadron_iso_real,
            N_hadron_iso_real=N_hadron_iso_real,
            photon_iso_real=photon_iso_real,
            pf_lists_real=pf_lists_real,
            
                x_fake=x_fake,y_fake=y_fake,w_fake=w_fake,pt_fake=pt_fake,bdt_fake=bdt_fake,
                C_hadron_iso_fake=C_hadron_iso_fake, N_hadron_iso_fake=N_hadron_iso_fake, photon_iso_fake=photon_iso_fake  , pf_lists_fake=pf_lists_fake    )
"""
#----------------------------------------------------------------------------------------------------------------------------------------
   
import multiprocessing

prefix = "root://xrootd-cms.infn.it//"
text_file = "Files_GJet_Pt-40toInf_DoubleEMEnriched_TuneCP5_13p6TeV_pythia8.txt"

def process_file(file_info):
    file_name, iteration = file_info
    print("Processing file:", file_name)
    main(file_iteration=iteration, path=prefix + file_name, distance=5)

if __name__ == '__main__':
    file_list = []
    with open(text_file, "r") as textfile:
        for i, line in enumerate(textfile):
            file_list.append((line.strip(), i))


    num_cores = multiprocessing.cpu_count()
    num_cores =2
    pool = multiprocessing.Pool(processes=num_cores)

    results = pool.map(process_file, file_list)

    pool.close()
    pool.join()



end = time.time()
print("The time passed is:", (end - start)/60," minutes")
