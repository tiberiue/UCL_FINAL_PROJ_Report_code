import uproot


#file = "root://eosuser.cern.ch//eos/user/j/jzahredd/JetTagging/final/Train_Test_split_lxplus/user.mykhando.27420846._000001.tree.root_train.root"
file = "/eos/user/j/jzahredd/JetTagging/final/Train_Test_split_lxplus/user.mykhando.27420846._000001.tree.root_train.root"
dsids = []
with uproot.open(file) as infile:
    tree = infile["FlatSubstructureJetTree"]
    dsids.append(tree["DSID"].array())

print (dsids)
