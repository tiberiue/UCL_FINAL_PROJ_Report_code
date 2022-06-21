import sys,os
import argparse

parser = argparse.ArgumentParser(description='Perform closure test.')

parser.add_argument('--model', dest='model', type=str,
                    required=True,
                    help='which Model to run on')

parser.add_argument('--train', dest='train', type=str,
                    required=True,
                    help='train, adv or comb')

parser.add_argument('--learningrate', dest='learningrate', type=float,
                    required= '--train' in sys.argv,
                    help='learning rate')

parser.add_argument('--nepochs', dest='nepochs', type=int,
                    required= '--train' in sys.argv,
                    help='nepochs')

parser.add_argument('--weight', dest='weight', type=int,
                    required=True,
                    help='without (0) or with (1) weight')

parser.add_argument('--nepochsadv', dest='nepochsadv', type=int,
                    required= '--train' in sys.argv,
                    help='nepochsadv')

parser.add_argument('--lambdaparam', dest='lambdaparam', type=int,
                    required= '--train' in sys.argv,
                    help='lambdaparam')

parser.add_argument('--lossparam', dest='lossparam', type=float,
                    required= '--train' in sys.argv,
                    help='lossparam')

#python adv.py --model LundNet --train adv --learningrate 0.0005 --weight 0 --nepochs 60 --nepochsadv 20 --lambdaparam 10 --lossparam 0.1

if __name__ == "__main__":

    args = parser.parse_args()

    PathToWorkArea = "/afs/cern.ch/work/j/jzahredd/LJPTagger/gitlab/ljptagger/"
    if args.weight == 0:
        f = open("{}src/configs/DifferentModels/config_{}_{}_nepochsadv{}_lambda{}_lossparam{}.yaml".format(PathToWorkArea, args.train, args.model, args.nepochsadv, args.lambdaparam, args.lossparam), "w")
    if args.weight == 1:
        f = open("{}src/configs/DifferentModels/config_{}_{}_nepochsadv{}_lambda{}_lossparam{}_weight.yaml".format(PathToWorkArea, args.train, args.model, args.nepochsadv, args.lambdaparam, args.lossparam), "w")


    ## Data
    f.write("data:\n")
    f.write("  path_to_trainfiles: \"/eos/user/j/jzahredd/JetTagging/final/Train_Test_split_lxplus/*_train.root\"\n")
    if args.weight == 1:
        f.write("  weights_file: \"/afs/cern.ch/work/j/jzahredd/LJPTagger/ljptagger/final/flat_weights_tr.root\"\n")
        f.write("  scale_factor: 14.475606\n")

    f.write("\n")

    ## architecture
    f.write("architecture:\n")
    f.write("  batch_size: 2058\n")
    f.write("  test_size: 0.2\n")
    f.write("  learning_rate: {}\n".format(args.learningrate))  #0.0005
    f.write("  num_gaussians: 20\n")
    f.write("  lambda_parameter: {}\n".format(args.lambdaparam))  #0.0005
    f.write("  loss_parameter: {}\n".format(args.lossparam))  #0.0005

    f.write("\n")

    ## classifier
    PathToModel = PathToWorkArea + "Models/" + "{}_lr{}_nepochs{}/".format(args.model, args.learningrate, args.nepochs)
    os.system("ls {}{}_e0{}*.pt | tail -n 1 > latest.txt".format(PathToModel, args.model, args.nepochs))

    with open("latest.txt") as fpt:
        for line in fpt:
            ckpt = line

    ckpt = ckpt.replace("\n", "")

    f.write("classifier:\n")
    f.write("  path_to_classifier_ckpt: \"{}\"\n".format(ckpt))
    f.write("  choose_model: {}\n".format(args.model)) ## LundNet, GATNet, GINNet, EdgeGinNet, PNANet

    f.write("\n")

    ## adversary
    f.write("adversary:\n")
    f.write("  save_adv_every_epoch: True\n")
    f.write("  adv_model_name: \"adv_lambda{}_lossparam{}_nepochadv{}_\"\n".format(args.lambdaparam, args.lossparam, args.nepochsadv))
    f.write("  n_epochs_adv: {}\n".format(args.nepochsadv))
    f.write("  path_to_store: \"{}\"\n".format(PathToModel))


    ## Script file
    if args.weight == 0:
        fsh = open("scripts/{}_{}_lambda{}_lossparam{}_nepochadv{}.sh".format(args.train, args.model, args.lambdaparam, args.lossparam, args.nepochsadv), "w")
    if args.weight == 1:
        fsh = open("scripts/{}_{}_lambda{}_lossparam{}_nepochadv{}_weight.sh".format(args.train, args.model, args.lambdaparam, args.lossparam, args.nepochsadv), "w")

    fsh.write("#!/bin/bash\n")
    fsh.write("cd /eos/user/j/jzahredd/JetTagging/\n")
    fsh.write("source miniconda3/bin/activate\n")
    fsh.write("conda activate rootenv\n")
    fsh.write("echo ${PWD}\n")
    fsh.write("cd /afs/cern.ch/work/j/jzahredd/LJPTagger/gitlab/ljptagger/src/\n")
    if args.weight == 0:
        fsh.write("python adv_train.py configs/DifferentModels/config_{}_{}_nepochsadv{}_lambda{}_lossparam{}.yaml".format(args.train, args.model, args.nepochsadv, args.lambdaparam, args.lossparam))
    if args.weight == 1:
        fsh.write("python weight_adv_train.py configs/DifferentModels/config_{}_{}_nepochsadv{}_lambda{}_lossparam{}_weight.yaml".format(args.train, args.model, args.nepochsadv, args.lambdaparam, args.lossparam))

    ## Submission file
    if args.weight == 0:
        fsubmit = open("scripts/{}_{}_lambda{}_lossparam{}_nepochadv{}.sub".format(args.train, args.model, args.lambdaparam, args.lossparam, args.nepochsadv), "w")
    if args.weight == 1:
        fsubmit = open("scripts/{}_{}_lambda{}_lossparam{}_nepochadv{}_weight.sub".format(args.train, args.model, args.lambdaparam, args.lossparam, args.nepochsadv), "w")

    fsubmit.write("# here goes your shell script\n")
    fsubmit.write("executable    = {}_{}_lambda{}_lossparam{}_nepochadv{}.sh\n".format(args.train, args.model, args.lambdaparam, args.lossparam, args.nepochsadv) if args.weight == 0 else "executable    = {}_{}_lambda{}_lossparam{}_nepochadv{}_weight.sh\n".format(args.train, args.model, args.lambdaparam, args.lossparam, args.nepochsadv))
    fsubmit.write("# here you specify where to put .log, .out and .err files\n")
    fsubmit.write("arguments               = $(ClusterId)$(ProcId)\n")
    fsubmit.write("output                  = debug/{}_{}_lambda{}_lossparam{}_nepochadv{}_w{}.$(ClusterId).out\n".format(args.train, args.model, args.lambdaparam, args.lossparam, args.nepochsadv, args.weight))
    fsubmit.write("error                   = debug/{}_{}_lambda{}_lossparam{}_nepochadv{}_w{}.$(ClusterId).err\n".format(args.train, args.model, args.lambdaparam, args.lossparam, args.nepochsadv, args.weight))
    fsubmit.write("log                     = debug/{}_{}_lambda{}_lossparam{}_nepochadv{}_w{}.$(ClusterId).log\n".format(args.train, args.model, args.lambdaparam, args.lossparam, args.nepochsadv, args.weight))
    fsubmit.write("should_transfer_files   = YES\n")
    fsubmit.write("transfer_input_files    = /afs/cern.ch/work/j/jzahredd/LJPTagger/gitlab/ljptagger/src/")
    fsubmit.write("adv_train.py\n" if args.weight == 0 else "weight_adv_train.py\n")
    fsubmit.write("when_to_transfer_output = ON_EXIT\n")
    fsubmit.write("request_GPUs = 1\n")
    fsubmit.write("request_CPUs = 1\n")
    fsubmit.write("requirements = regexp(\"V100\", TARGET.CUDADeviceName)\n")
    fsubmit.write("#requirements = TARGET.CUDADriverVersion=?= 11.1\n")
    fsubmit.write("+JobFlavour = \"nextweek\"\n")
    fsubmit.write("queue\n")
