## Classifier training
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


## Adversarial training
python adv.py --model LundNet    --train adv --learningrate 0.0005 --weight 0 --nepochs 60 --nepochsadv 20 --lambdaparam 10 --lossparam 0.1
python adv.py --model GATNet     --train adv --learningrate 0.0005 --weight 0 --nepochs 60 --nepochsadv 20 --lambdaparam 10 --lossparam 0.1
python adv.py --model GINNet     --train adv --learningrate 0.0005 --weight 0 --nepochs 60 --nepochsadv 20 --lambdaparam 10 --lossparam 0.1
python adv.py --model EdgeGinNet --train adv --learningrate 0.0005 --weight 0 --nepochs 60 --nepochsadv 20 --lambdaparam 10 --lossparam 0.1
python adv.py --model PNANet     --train adv --learningrate 0.0005 --weight 0 --nepochs 60 --nepochsadv 20 --lambdaparam 10 --lossparam 0.1

python adv.py --model LundNet    --train adv --learningrate 0.0005 --weight 1 --nepochs 60 --nepochsadv 20 --lambdaparam 10 --lossparam 0.1
python adv.py --model GATNet     --train adv --learningrate 0.0005 --weight 1 --nepochs 60 --nepochsadv 20 --lambdaparam 10 --lossparam 0.1
python adv.py --model GINNet     --train adv --learningrate 0.0005 --weight 1 --nepochs 60 --nepochsadv 20 --lambdaparam 10 --lossparam 0.1
python adv.py --model EdgeGinNet --train adv --learningrate 0.0005 --weight 1 --nepochs 60 --nepochsadv 20 --lambdaparam 10 --lossparam 0.1
python adv.py --model PNANet     --train adv --learningrate 0.0005 --weight 1 --nepochs 60 --nepochsadv 20 --lambdaparam 10 --lossparam 0.1


## Combined training
python comb.py --model LundNet    --train comb --learningrate 0.0005 --weight 0 --nepochs 60 --nepochsadv 20 --lambdaparam 10 --lossparam 0.1 --lrratio 0.001 --lroptimizer 0.0005 --nepochscommon 200
python comb.py --model GATNet     --train comb --learningrate 0.0005 --weight 0 --nepochs 60 --nepochsadv 20 --lambdaparam 10 --lossparam 0.1 --lrratio 0.001 --lroptimizer 0.0005 --nepochscommon 200
python comb.py --model GINNet     --train comb --learningrate 0.0005 --weight 0 --nepochs 60 --nepochsadv 20 --lambdaparam 10 --lossparam 0.1 --lrratio 0.001 --lroptimizer 0.0005 --nepochscommon 200
python comb.py --model EdgeGinNet --train comb --learningrate 0.0005 --weight 0 --nepochs 60 --nepochsadv 20 --lambdaparam 10 --lossparam 0.1 --lrratio 0.001 --lroptimizer 0.0005 --nepochscommon 200
python comb.py --model PNANet     --train comb --learningrate 0.0005 --weight 0 --nepochs 60 --nepochsadv 20 --lambdaparam 10 --lossparam 0.1 --lrratio 0.001 --lroptimizer 0.0005 --nepochscommon 200

python comb.py --model LundNet    --train comb --learningrate 0.0005 --weight 1 --nepochs 60 --nepochsadv 20 --lambdaparam 10 --lossparam 0.1 --lrratio 0.001 --lroptimizer 0.0005 --nepochscommon 200
python comb.py --model GATNet     --train comb --learningrate 0.0005 --weight 1 --nepochs 60 --nepochsadv 20 --lambdaparam 10 --lossparam 0.1 --lrratio 0.001 --lroptimizer 0.0005 --nepochscommon 200
python comb.py --model GINNet     --train comb --learningrate 0.0005 --weight 1 --nepochs 60 --nepochsadv 20 --lambdaparam 10 --lossparam 0.1 --lrratio 0.001 --lroptimizer 0.0005 --nepochscommon 200
python comb.py --model EdgeGinNet --train comb --learningrate 0.0005 --weight 1 --nepochs 60 --nepochsadv 20 --lambdaparam 10 --lossparam 0.1 --lrratio 0.001 --lroptimizer 0.0005 --nepochscommon 200
python comb.py --model PNANet     --train comb --learningrate 0.0005 --weight 1 --nepochs 60 --nepochsadv 20 --lambdaparam 10 --lossparam 0.1 --lrratio 0.001 --lroptimizer 0.0005 --nepochscommon 200


## Scores
python scores.py --model LundNet    --train score --learningrate 0.0005 --nepochs 60 --weight 0
python scores.py --model GATNet     --train score --learningrate 0.0005 --nepochs 60 --weight 0
python scores.py --model GINNet     --train score --learningrate 0.0005 --nepochs 60 --weight 0
python scores.py --model EdgeGinNet --train score --learningrate 0.0005 --nepochs 60 --weight 0
python scores.py --model PNANet     --train score --learningrate 0.0005 --nepochs 60 --weight 0

python scores.py --model LundNet    --train score --learningrate 0.0005 --nepochs 60 --weight 1
python scores.py --model GATNet     --train score --learningrate 0.0005 --nepochs 60 --weight 1
python scores.py --model GINNet     --train score --learningrate 0.0005 --nepochs 60 --weight 1
python scores.py --model EdgeGinNet --train score --learningrate 0.0005 --nepochs 60 --weight 1
python scores.py --model PNANet     --train score --learningrate 0.0005 --nepochs 60 --weight 1
