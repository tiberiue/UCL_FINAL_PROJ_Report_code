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

# python scores.py --model LundNet --train score --learningrate 0.0005 --nepochs 60 --weight 0

if __name__ == "__main__":

    args = parser.parse_args()

    PathToWorkArea = "/afs/cern.ch/work/j/jzahredd/LJPTagger/gitlab/ljptagger/"
    if args.weight == 0:
        f = open("{}src/configs/DifferentModels/config_{}_{}_lr{}_nepochs{}.yaml".format(PathToWorkArea, args.train, args.model, args.learningrate, args.nepochs), "w")
    if args.weight == 1:
        f = open("{}src/configs/DifferentModels/config_{}_{}_lr{}_nepochs{}_weight.yaml".format(PathToWorkArea, args.train, args.model, args.learningrate, args.nepochs), "w")


    ## Data
    f.write("data:\n")
    f.write("  path_to_test_file: \"/eos/user/j/jzahredd/JetTagging/final/Train_Test_split_lxplus/*_test.root\"\n")
    if args.weight == 0:
        os.system("mkdir {}Scores/{}_lr{}_nepochs{}".format(PathToWorkArea, args.model, args.learningrate, args.nepochs))
        f.write("  path_to_outdir: \"{}Scores/{}_lr{}_nepochs{}/\"\n".format(PathToWorkArea, args.model, args.learningrate, args.nepochs))
    if args.weight == 1:
        os.system("mkdir {}Scores/{}_lr{}_nepochs{}_weight".format(PathToWorkArea, args.model, args.learningrate, args.nepochs))
        f.write("  path_to_outdir: \"{}Scores/{}_lr{}_nepochs{}_weight/\"\n".format(PathToWorkArea, args.model, args.learningrate, args.nepochs))

    f.write("\n")

    ## architecture
    PathToModel = PathToWorkArea + "Models/" + "{}_lr{}_nepochs{}/".format(args.model, args.learningrate, args.nepochs)
    os.system("ls {}combined_class*.pt | tail -n 1 > latest.txt".format(PathToModel))

    with open("latest.txt") as fpt:
        for line in fpt:
            ckpt = line

    ckpt = ckpt.replace("\n", "")

    f.write("test:\n")
    f.write("  path_to_combined_ckpt: \"{}\"\n".format(ckpt))
    f.write("  output_name: \"{}Scores\"\n".format(args.model))
    f.write("  choose_model: {}\n".format(args.model))
    f.write("  learning_rate: {}\n".format(args.learningrate))
    f.write("  batch_size: 2048\n")  #0.0005


    ## Script file
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
        fsh.write("python make_scores.py configs/DifferentModels/config_{}_{}_lr{}_nepochs{}.yaml".format(args.train, args.model, args.learningrate, args.nepochs))
    if args.weight == 1:
        fsh.write("python make_scores.py configs/DifferentModels/config_{}_{}_lr{}_nepochs{}_weight.yaml".format(args.train, args.model, args.learningrate, args.nepochs))


    ## Submission file
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
    fsubmit.write("transfer_input_files    = /afs/cern.ch/work/j/jzahredd/LJPTagger/gitlab/ljptagger/src/make_scores.py\n")
    fsubmit.write("when_to_transfer_output = ON_EXIT\n")
    fsubmit.write("request_GPUs = 1\n")
    fsubmit.write("request_CPUs = 1\n")
    fsubmit.write("requirements = regexp(\"V100\", TARGET.CUDADeviceName)\n")
    fsubmit.write("#requirements = TARGET.CUDADriverVersion=?= 11.1\n")
    fsubmit.write("+JobFlavour = \"nextweek\"\n")
    fsubmit.write("queue\n")
