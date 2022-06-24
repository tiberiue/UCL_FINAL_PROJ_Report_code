import uproot
import glob

path_to_file = "root://eosuser.cern.ch//eos/user/j/jzahredd/JetTagging/final/Train_Test_split_lxplus/user.mykhando.*_train.root"
files = glob.glob(path_to_file)

#file = "/eos/user/j/jzahredd/JetTagging/final/Train_Test_split_lxplus/user.mykhando.27420846._000001.tree.root_train.root"

dsids = []
with uproot.open(file) as infile:
    tree = infile["FlatSubstructureJetTree"]
    dsids.append(tree["DSID"].array())

n_epochs = 2
for epoch in range(n_epochs):
    metrics_filename = "root://eosuser.cern.ch//eos/user/j/jzahredd/JetTagging/losses_"+str(i)+".txt"
    f = open(metrics_filename, "w")
    f.write(epoch)
    f.close()


print (dsids)
