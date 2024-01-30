#!/usr/bin/env python

from utils_plots import roc_from_histos
from utils_plots import dijet_xsweights_dict
from utils_plots import assign_weights
from utils_plots import make_rocs
from utils_plots import make_efficiencies
from utils_plots import make_efficiencies_pt
from utils_plots import make_efficiencies_3var
from utils_plots import mass_sigeff
from utils_plots import get_wp_th1
from utils_plots import mass_bgrej
from utils_plots import get_eff_score
from utils_plots import wp50_cut
from utils_plots import get_wp_tag
from utils_plots import tagger_scores
from utils_plots import scores_separation, scores_separation_pt
from utils_plots import mass_sculpting
from utils_plots import mass_sculpting_ptcut
from utils_plots import pt_spectrum
from utils_plots import pt_bgrej
from utils_plots import pt_sigeff
from utils_plots import weights
from utils_plots import trivar_scores
from utils_plots import getANNROCresults
from utils_plots import JSD
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import entropy
import numpy as np
import ROOT
from root_numpy import fill_hist  as fh
import warnings
warnings.filterwarnings('ignore')


taggers = {}
tagger_files = {}
#tagger_files["LundNet_class"] = "/Users/mykola/Physics/LundTagger/LScores/scores_newclass.root"
#tagger_files["LundNet_adv059"] = "/Users/mykola/Physics/LundTagger/LScores/scores_adv_0.59639.root"
tagger_files["LundNet_class"] = "/eos/user/t/tmlinare/Lund_tagger/ljptagger/Models/score/user.teris.1._000001.tree.root_test.root_score_LundNetScores.root"
#tagger_files["LundNet_class"] = "/eos/home-m/mykhando/public/scores_LundNet_class.root"
#tagger_files["LundNet_class"] = "/afs/cern.ch/work/j/jzahredd/LJPTagger/gitlab/ljptagger/Scores/LundNet_lr0.0005_nepochs60/tree.root"
#tagger_files["LundNet_class"] = "/afs/cern.ch/work/j/jzahredd/LJPTagger/gitlab/ljptagger/Scores/user.mykhando.27420846._000001.tree.root_test.root_score_LundNetScores.root "
#tagger_files["3var"] = "/eos/user/m/mykhando/public/scores_3var.root"

working_point = 0.5
for t in tagger_files:
    taggers[t] = tagger_scores(t,tagger_files[t], working_point)


for t in taggers:
    if taggers[t].name == "3var":
        continue
    get_wp_th1(taggers[t],working_point)

#######################################################################################
#######################################################################################
#pt_sigeff(taggers,"no_weight")
#mass_sigeff(taggers,"no_weight")  ## fixed
#mass_sigeff(taggers,"flat_weight")
#mass_sigeff(taggers,"xsec_weight")
#mass_sigeff(taggers)
#mass_bgrej(taggers)
#mass_bgrej(taggers,"no_weight")
#mass_bgrej(taggers,"xsec_weight")
#mass_bgrej(taggers,"flat_weight")
#######################################################################################
#######################################################################################

## weights: no_weight, flat_weight, xsec_weight, chris_weight
## Make roc curves plot
make_rocs(taggers)  ## fixed

## Make plot pT spectrum
pt_spectrum(taggers, weight="chris_weight") ## fixed

## Make plot background rejection vs pT
pt_bgrej(taggers, weight="chris_weight") ## fixed

## Make mass sculpting plots (inclusive and in bins of pT)
for t in taggers:
    mass_sculpting(taggers[t], weight="chris_weight")  ## fixed
    mass_sculpting_ptcut(taggers[t],  200,  500, weight="chris_weight")  ## fixed
    mass_sculpting_ptcut(taggers[t],  500, 1000, weight="chris_weight")  ## fixed
    mass_sculpting_ptcut(taggers[t], 1000, 3000, weight="chris_weight")  ## fixed

## Make background rejection vs signal efficiency plots (inclusive and in bins of pT)
make_efficiencies_3var(taggers) ## fixed
make_efficiencies_pt(taggers,  200,  500, weight="chris_weight") ## fixed
make_efficiencies_pt(taggers,  500, 1000, weight="chris_weight") ## fixed
make_efficiencies_pt(taggers, 1000, 3000, weight="chris_weight") ## fixed

## Make classification plots
for t in taggers:
    if taggers[t].name == "3var":
        continue
    scores_separation(taggers[t]) ## fixed
    scores_separation_pt(taggers[t],  200,  500) ## fixed
    scores_separation_pt(taggers[t],  500, 1000) ## fixed
    scores_separation_pt(taggers[t], 1000, 3000) ## fixed
