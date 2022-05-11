
import glob
import uproot3



#files = glob.glob("/eos/user/r/riordan/data/wprime/*_train.root") 
#files = glob.glob("/eos/user/r/riordan/data/j3to9/*_train.root")
files = glob.glob("/sps/atlas/k/khandoga/Scores/*.root")

#files -= glob.glob("/eos/user/m/mykhando/public/FlatSamples/Train_Test_split_lxplus/*27245469*train*.root")

print ("files:",files)
#intreename = "lundjets_InDetTrackParticles"
intreename = "FlatSubstructureJetTree"

nentries_total = 0

for file in files:
    nentries_total += uproot3.numentries(file, intreename)

print("Evaluating on {} files with {} entries in total.".format(len(files), nentries_total))
