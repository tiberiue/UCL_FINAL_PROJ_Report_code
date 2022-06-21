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

if __name__ == "__main__":


    args = parser.parse_args()

    if args.weight == 0:
        f = open("/afs/cern.ch/work/j/jzahredd/LJPTagger/gitlab/ljptagger/src/configs/DifferentModels/config_{}_{}_lr{}_nepochs{}.yaml".format(args.train, args.model, args.learningrate, args.nepochs), "w")
    if args.weight == 1:
        f = open("/afs/cern.ch/work/j/jzahredd/LJPTagger/gitlab/ljptagger/src/configs/DifferentModels/config_{}_{}_lr{}_nepochs{}_weight.yaml".format(args.train, args.model, args.learningrate, args.nepochs), "w")


    f.write("data:\n")
    f.write("  path_to_trainfiles: \"/eos/user/j/jzahredd/JetTagging/final/Train_Test_split_lxplus/*_train.root\"\n")  # /sps/atlas/k/khandoga/MySamplesS40/user.mykhando.27366163._000001.tree.root_train.root
    #path_to_output = args.model + "_" + args.learningrate + "_" + args.nepochs
    if args.weight == 0:
        os.system("mkdir /afs/cern.ch/work/j/jzahredd/LJPTagger/gitlab/ljptagger/Models/{}_lr{}_nepochs{}".format(args.model, args.learningrate, args.nepochs))
    if args.weight == 1:
        os.system("mkdir /afs/cern.ch/work/j/jzahredd/LJPTagger/gitlab/ljptagger/Models/{}_lr{}_nepochs{}_weight".format(args.model, args.learningrate, args.nepochs))

    if args.weight == 0:
        f.write("  path_to_save: \"/afs/cern.ch/work/j/jzahredd/LJPTagger/gitlab/ljptagger/Models/{}_lr{}_nepochs{}/"\n".format(args.model, args.learningrate, args.nepochs))
    if args.weight == 1:
        f.write("  path_to_save: \"/afs/cern.ch/work/j/jzahredd/LJPTagger/gitlab/ljptagger/Models/{}_lr{}_nepochs{}_weight/"\n".format(args.model, args.learningrate, args.nepochs))

    f.write("  model_name: \"{}_\"\n".format(args.model) if args.weight == 0 else "  model_name: \"{}_weighted_\"\n".format(args.model))
    if args.weight == 1:
        f.write("  weights_file: \"/afs/cern.ch/work/j/jzahredd/LJPTagger/ljptagger/final/flat_weights_tr.root\"\n")
        f.write("  scale_factor: 14.475606\n")
    f.write("\n")
    f.write("architecture:\n")
    f.write("  batch_size: 2048\n")
    f.write("  test_size: 0.2\n")
    f.write("  n_epochs: {}\n".format(args.nepochs)) # 60
    f.write("  learning_rate: {}\n".format(args.learningrate))  #0.0005
    f.write("  choose_model: {}\n".format(args.model)) ## LundNet, GATNet, GINNet, EdgeGinNet, PNANet
    f.write("  save_every_epoch: True\n")
    f.write("\n")

    f.write("retrain:\n")
    f.write("  flag: False\n")
    f.write("  path_to_ckpt: \"/sps/atlas/k/khandoga/TrainGNN/Models/LundNet_ufob_e012_0.12481.pt\"\n")

    if args.weight == 0:
        fsh = open("scripts/{}_{}_lr{}_nepochs{}.sh".format(args.train, args.model, args.learningrate, args.nepochs), "w")
    if args.weight == 1:
        fsh = open("scripts/{}_{}_lr{}_nepochs{}_weight.sh".format(args.train, args.model, args.learningrate, args.nepochs), "w")

    fsh.write("#!/bin/bash\n")
    fsh.write("cd /eos/user/j/jzahredd/JetTagging/\n")
    fsh.write("source miniconda3/bin/activate\n")
    fsh.write("conda activate rootenv\n")
    fsh.write("echo ${PWD}\n")
    fsh.write("cd /afs/cern.ch/work/j/jzahredd/LJPTagger/gitlab/ljptagger/src/\n")
    if args.weight == 0:
        fsh.write("python class_train.py configs/DifferentModels/config_{}_{}_lr{}_nepochs{}.yaml".format(args.train, args.model, args.learningrate, args.nepochs))
    if args.weight == 1:
        fsh.write("python weight_class_train.py configs/DifferentModels/config_{}_{}_lr{}_nepochs{}_weight.yaml".format(args.train, args.model, args.learningrate, args.nepochs))

    if args.weight == 0:
        fsubmit = open("scripts/{}_{}_lr{}_nepochs{}.sub".format(args.train, args.model, args.learningrate, args.nepochs), "w")
    if args.weight == 1:
        fsubmit = open("scripts/{}_{}_lr{}_nepochs{}_weight.sub".format(args.train, args.model, args.learningrate, args.nepochs), "w")

    fsubmit.write("# here goes your shell script\n")
    fsubmit.write("executable    = {}_{}_lr{}_nepochs{}.sh\n".format(args.train, args.model, args.learningrate, args.nepochs) if args.weight == 0 else "executable    = {}_{}_lr{}_nepochs{}_weight.sh\n".format(args.train, args.model, args.learningrate, args.nepochs))
    fsubmit.write("# here you specify where to put .log, .out and .err files\n")
    fsubmit.write("arguments               = $(ClusterId)$(ProcId)\n")
    fsubmit.write("output                  = debug/{}_{}_lr{}_nepochs{}_w{}.$(ClusterId).out\n".format(args.train, args.model, args.learningrate, args.nepochs, args.weight))
    fsubmit.write("error                   = debug/{}_{}_lr{}_nepochs{}_w{}.$(ClusterId).err\n".format(args.train, args.model, args.learningrate, args.nepochs, args.weight))
    fsubmit.write("log                     = debug/{}_{}_lr{}_nepochs{}_w{}.$(ClusterId).log\n".format(args.train, args.model, args.learningrate, args.nepochs, args.weight))
    fsubmit.write("should_transfer_files   = YES\n")
    fsubmit.write("transfer_input_files    = /afs/cern.ch/work/j/jzahredd/LJPTagger/gitlab/ljptagger/src/")
    fsubmit.write("class_train.py\n" if args.weight == 0 else "weight_class_train.py\n")
    fsubmit.write("when_to_transfer_output = ON_EXIT\n")
    fsubmit.write("request_GPUs = 1\n")
    fsubmit.write("request_CPUs = 1\n")
    fsubmit.write("requirements = regexp(\"V100\", TARGET.CUDADeviceName)\n")
    fsubmit.write("#requirements = TARGET.CUDADriverVersion=?= 11.1\n")
    fsubmit.write("+JobFlavour = \"nextweek\"\n")
    fsubmit.write("queue\n")

'''
python train.py --model LundNet --train train --learningrate 0.0005 --weight 0 --nepochs 60
python train.py --model GATNet --train train --learningrate 0.0005 --weight 0 --nepochs 60
python train.py --model GINNet --train train --learningrate 0.0005 --weight 0 --nepochs 60
python train.py --model EdgeGinNet --train train --learningrate 0.0005 --weight 0 --nepochs 60
python train.py --model PNANet --train train --learningrate 0.0005 --weight 0 --nepochs 60

python train.py --model LundNet --train train --learningrate 0.0005 --weight 1 --nepochs 60
python train.py --model GATNet --train train --learningrate 0.0005 --weight 1 --nepochs 60
python train.py --model GINNet --train train --learningrate 0.0005 --weight 1 --nepochs 60
python train.py --model EdgeGinNet --train train --learningrate 0.0005 --weight 1 --nepochs 60
python train.py --model PNANet --train train --learningrate 0.0005 --weight 1 --nepochs 60

#python train.py --model LundNet --train train --learningrate 0.001 --weight 0 --nepochs 60
#python train.py --model GATNet --train train --learningrate 0.001 --weight 0 --nepochs 60
#python train.py --model GINNet --train train --learningrate 0.001 --weight 0 --nepochs 60
#python train.py --model EdgeGinNet --train train --learningrate 0.001 --weight 0 --nepochs 60
#python train.py --model PNANet --train train --learningrate 0.001 --weight 0 --nepochs 60

#python train.py --model LundNet --train train --learningrate 0.001 --weight 1 --nepochs 60
#python train.py --model GATNet --train train --learningrate 0.001 --weight 1 --nepochs 60
#python train.py --model GINNet --train train --learningrate 0.001 --weight 1 --nepochs 60
#python train.py --model EdgeGinNet --train train --learningrate 0.001 --weight 1 --nepochs 60
#python train.py --model PNANet --train train --learningrate 0.001 --weight 1 --nepochs 60

condor_submit train_EdgeGinNet_lr0.0005_nepochs60.sub
condor_submit train_EdgeGinNet_lr0.0005_nepochs60_weight.sub

condor_submit train_LundNet_lr0.0005_nepochs60.sub
condor_submit train_LundNet_lr0.0005_nepochs60_weight.sub

condor_submit train_PNANet_lr0.0005_nepochs60.sub
condor_submit train_PNANet_lr0.0005_nepochs60_weight.sub

condor_submit train_GATNet_lr0.0005_nepochs60.sub
condor_submit train_GATNet_lr0.0005_nepochs60_weight.sub

condor_submit train_GINNet_lr0.0005_nepochs60.sub
condor_submit train_GINNet_lr0.0005_nepochs60_weight.sub

'''
