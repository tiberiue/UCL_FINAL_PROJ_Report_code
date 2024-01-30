#import uproot
#import awkward as ak
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ROOT import TFile, TH1D, TH2D, TH2F, TH1F, TCanvas, TGraph, TMultiGraph, TLegend, TColor
import math
import pandas as pd
from root_numpy import fill_hist  as fh
from root_numpy import array2hist as a2h
from root_numpy import hist2array as h2a
from root_numpy import tree2array
from scipy.stats import entropy
from rootplotting import ap
from rootplotting.tools import *

MASSBINS = np.linspace(200, 3000, (300 - 40) // 5 + 1, endpoint=True)

def roc_from_histos(h_sig, h_bkg, wpcut=None):
    """
    Compute a ROC curve given two histograms.
    """

    tprs = []
    fprs = []

    #num_bins = h_sig.GetXaxis().GetNbins()
    num_bins = h_sig.GetNbinsX()
    n_sig = h_sig.Integral(1,num_bins+1)
    n_bkg = h_bkg.Integral(1,num_bins+1)

    if wpcut:
        i_wpbin = h_sig.GetXaxis().FindBin(wpcut)

    tpr_wp = 0
    fpr_wp = 0

    for i_bin in range(1, num_bins+2):

        n_tp = h_sig.Integral(i_bin, num_bins+1)
        n_fp = h_bkg.Integral(i_bin, num_bins+1)

 #       print("Bin no {}: ntp = {}, nfp = {}".format(i_bin, n_tp, n_fp))

        tpr = n_tp / n_sig
        fpr = n_fp / n_bkg

 #       print("Bin no {}: FPR = {}, TPR = {}".format(i_bin, fpr, tpr))

        tprs.append(tpr)
        fprs.append(fpr)

        if wpcut and i_bin == i_wpbin:
 #           print("score bin no at wp =", i_bin)
            tpr_wp = tpr
            fpr_wp = fpr
 #           print("tpr_wp = {:.2f}/{:.2f}={:.2f}".format(n_tp,n_sig,tpr))
 #           print("fpr_wp = {:.2f}/{:.2f}={:.2f}".format(n_fp,n_bkg,fpr))


    a_tprs = np.array(tprs)
    a_fprs = np.array(fprs)

    #get area under curve
    auc = np.abs( np.trapz(a_tprs, a_fprs) )

    #flip curve if auc is negative
    if auc < 0.5:
        a_tprs = 1 - a_tprs
        a_fprs = 1 - a_fprs
        auc = 1 - auc

        if wpcut:
            tpr_wp = 1 - tpr_wp
            fpr_wp = 1 - fpr_wp

    if wpcut:
        return a_tprs, a_fprs, auc, tpr_wp, fpr_wp

    else:
        return a_tprs, a_fprs, auc


dijet_xsweights_dict = {
    361022:   811423.536 *    1.0,
    361023:   8453.64024 *    1.0,
    361024:   134.9920945 *   1.0,
    361025:   4.19814486 *    1.0,
    361026:   0.241941709 *   1.0,
    361027:   0.006358874 *   1.0,
    361028:   0.006354782 *   1.0,
    361029:   0.000236819 *   1.0,
    361030:   7.054e-06 *     1.0,
    361031:   1.13e-07 *      1.0,
    361032:   4.405975e-10 *  1.0,

    364702: 2433000 * 0.0098631 / 1110002,
    364703: 26450 * 0.011658 / 1671907,
    364704: 254.61 * 0.013366 / 1839956,
    364705: 4.5529 * 0.014526 / 1435042,  # This is the only thing that changed due to extra root file for full
    364706: 0.25754 * 0.0094734 / 773626, # declustering
    364707: 0.016215 * 0.011097 / 960798,
    364708: 0.00062506 * 0.010156 / 1315619,
    364709: 1.9639E-05 * 0.012056 / 1082543,
    364710: 1.1962E-06 * 0.0058933 / 201761,
    364711: 4.2263E-08 * 0.002673 / 247582,
    364712: 1.0367E-09 * 0.00042889 / 288782,

    #ttbar
    426347:   1.0,
    426345:   1.0,
    -1: 1.0,
}


def assign_weights(mcid, mcweight):
    return dijet_xsweights_dict[mcid]*mcweight

def make_rocs(taggers):
    colours = [ROOT.kViolet + 7, ROOT.kAzure + 7, ROOT.kTeal, ROOT.kSpring - 2, ROOT.kOrange - 3, ROOT.kPink,  ROOT.kPink+3]

    c1 = ap.canvas(num_pads=1, batch=True)
    count = 0
    for t in taggers:
        if taggers[t].name == "3var":
            continue
        tprs, fprs, auc = taggers[t].get_roc()

        ## Fill random guessing
        if count==0:
            h = TGraph(len(tprs), np.linspace(0.0001, 1, len(tprs)), np.linspace(0.0001, 1, len(tprs)))
            c1.graph(h, linestyle=2, linecolor=ROOT.kBlack, option="AL", label="Random guessing")

        h = TGraph(len(tprs), fprs, tprs)
        c1.graph(h, linestyle=1, linecolor=colours[count], markercolor=colours[count], option="L", label=taggers[t].name)
        count+=1

    c1.xlabel('False positive rate')
    c1.ylabel('True positive rate')
    c1.xlim(0, 1) ## c1.xlim(0.2, 1)
    #c1.ylim(0, 1.2)

    c1.text(["#sqrt{s} = 13 TeV, #it{W} tagging",
            "anti-k_{t} R=1.0 UFO jets", 'AUC:'+str(auc), 
            ], qualifier='Simulation Internal')
    c1.legend()
    c1.save("/eos/user/t/tmlinare/Lund_tagger/ljptagger/Models/Graphs/ROCcurves.png")



def make_efficiencies_3var(taggers):

    colours = [ROOT.kViolet + 7, ROOT.kAzure + 7, ROOT.kTeal, ROOT.kSpring - 2, ROOT.kOrange - 3, ROOT.kPink,  ROOT.kPink+3]

    count = 0
    c1 = ap.canvas(num_pads=1, batch=True)
    for t in taggers:
        if taggers[t].name == "3var":
            continue


        tprs, fprs, auc = taggers[t].get_roc()
        ## protect for inf
        inv = 1/fprs
        inv[inv == np.inf] = 1e10

        ## Fill random guessing
        h = TGraph(len(tprs), np.linspace(0.0001, 1, len(tprs)), 1/np.linspace(0.0001, 1, len(tprs)))
        c1.graph(h, linestyle=2, linecolor=ROOT.kBlack, option="AL", label="Random guessing")

        ## create TGraph for each tagger
        h = TGraph(len(tprs), tprs, inv)
        c1.graph(h, linestyle=1, linecolor=colours[count], markercolor=colours[count], markerstyle=1, option="L", label=taggers[t].name)

        ## Plot other things too
        if count == len(taggers)-1:
            ## Fill DNN (Davide)
            h = TGraph(len(getANNROCresults("x", "NN")), getANNROCresults("x", "NN"), getANNROCresults("y", "NN"))
            c1.graph(h, linestyle=1, linecolor=ROOT.kRed, markercolor=ROOT.kRed, markerstyle=1, option="L", label="z_{NN}")
            ## Fill ANN (Davide)
            h = TGraph(len(getANNROCresults("x", "ANN")), getANNROCresults("x", "ANN"), getANNROCresults("y", "ANN"))
            c1.graph(h, linestyle=1, linecolor=ROOT.kBlue, markercolor=ROOT.kBlue, markerstyle=1, option="L", label="z_{ANN}^{#lambda=10}")
            ## Fill 3-var tagger
            h = TGraph()
            h.SetPoint(0, 0.5072431, 40.86972)
            c1.graph(h, linecolor=ROOT.kWhite, markercolor=ROOT.kRed, markerstyle=20, option="P", label="3-var tagger")

        count+=1


    c1.xlabel('Signal efficiency')
    c1.ylabel('Background rejection')
    c1.xlim(0.1, 1) ## c1.xlim(0.2, 1)
    c1.ylim(1, 1e7)

    c1.text(["#sqrt{s} = 13 TeV, #it{W} tagging",
            "anti-k_{t} R=1.0 UFO jets",
            ], qualifier='Simulation Internal')
    c1.log()
    c1.legend()
    c1.save("/eos/user/t/tmlinare/Lund_tagger/ljptagger/Models/Graphs/Efficiencies_3var_taggers.png")


def make_efficiencies(taggers):

    plt.figure(figsize=(16,12))

    for t in taggers:
        if taggers[t].name == "3var":
            continue
        tprs, fprs, auc = taggers[t].get_roc()
        plt.semilogy(tprs, 1/fprs, label="{0}".format(taggers[t].name))

    plt.semilogy(np.linspace(0, 1, 100),1/np.linspace(0, 1, 100),'k--',label="Random guessing")
    plt.xlim(0.0, 1.0)
    plt.ylim(1, 1e8)
    plt.title("QCD rejection vs. W tagging efficiency")
    plt.xlabel("Signal efficiency")
    plt.ylabel("Background rejection")

    #tprs, fprs, auc = roc_from_histos(h_signal, h_bg)

    plt.legend()
    #plt.savefig("w_cor_uncor1",dpi=500)
    plt.show()

def get_eff_score(pt_vs_score,wp):
    scores_projection = pt_vs_score.ProjectionX()
    pt_value = []
    tag_score = []
    for ptbin in range(1,pt_vs_score.GetNbinsX()+1):
        curcont = 0
        #pt_value.append(scores_projection.GetBinCenter(ptbin))
        if scores_projection.GetBinContent(ptbin)==0:
            tag_score.append(0)
            pt_value.append(scores_projection.GetBinCenter(ptbin))
            continue
        #for scorebin in range(1, pt_vs_score.GetNbinsX()):
        for scorebin in range(1, pt_vs_score.GetNbinsY()+1):
            curcont += pt_vs_score.GetBinContent(ptbin, scorebin)
            if curcont/scores_projection.GetBinContent(ptbin) >= wp:
                tag_score.append(scorebin/100.)
                pt_value.append(scores_projection.GetBinCenter(ptbin))
                break
    return pt_value, tag_score


def wp50_cut(p,pt):
    return p[0]+p[1]/(p[2]+math.exp(p[3]*(pt+p[4])))


def get_wp_tag(tagger, wp):
    h_pt_nn   = TH2D( "h_pt_nn{}".format(tagger.name), "h_pt_nn{}".format(tagger.name), 100, 0., 3000,100,0,1 )
    for pt,nn in zip(tagger.signal["fjet_pt"],tagger.signal["fjet_nnscore"]):
        h_pt_nn.Fill(pt,nn)
    pts, scores = get_eff_score(h_pt_nn,wp)
    scores = scores[6:]
    pts = pts [6:]
    gra = TGraph(len(pts), np.array(pts).astype("float"), np.array(scores).astype("float"))
    fitfunc = TF1("fit", "[p0]+[p1]/([p2]+exp([p3]*(x+[p4])))", 200, 2700) #exponential sigmoid fit (best so far)
    #fitfunc = root.TF1("fit", "pol10", 200, 2700) #12th order polynomial fit
    gra.Fit(fitfunc,"R,S")
    c = TCanvas("myCanvasName{}".format(tagger.name),"The Canvas Title{}",800,600)

    gra.Draw()

    p = fitfunc.GetParameters()
    tagger.scores["tag_cut"] = np.vectorize(lambda x:p[0]+p[1]/(p[2]+math.exp(p[3]*(x+p[4]))))(tagger.scores.fjet_pt)
    tagger.signal = tagger.scores[tagger.scores.EventInfo_mcChannelNumber>370000]
    tagger.bg = tagger.scores[tagger.scores.EventInfo_mcChannelNumber<370000]

    tagger.bg_tagged = tagger.bg[tagger.bg.fjet_nnscore > tagger.bg.tag_cut]
    tagger.bg_untagged = tagger.bg[tagger.bg.fjet_nnscore < tagger.bg.tag_cut]
    tagger.signal_tagged = tagger.signal[tagger.signal.fjet_nnscore > tagger.signal.tag_cut]
    c.Draw()

def get_wp_th1(tagger,wp):
    print("I am here in get_wp_th1 ")
    print (tagger.name)
    h_pt_nn     = TH2D( "h_pt_nn{}".format(tagger.name), "h_pt_nn{}".format(tagger.name), 100, 0., 3000,100,0,1 )
    h_pt_nn.SetDirectory(0)
    for pt,nn in zip(tagger.signal["fjet_pt"].values,tagger.signal["fjet_nnscore"].values):
        h_pt_nn.Fill(pt,nn)

    c = TCanvas("", "", 500, 500)
    c.SetRightMargin(0.2)
    h_pt_nn.Draw("colz")
    c.SaveAs("/eos/user/t/tmlinare/Lund_tagger/ljptagger/Models/Graphs/pt_vs_score.png")

    pts, scores = get_eff_score(h_pt_nn,wp)
    print(len(pts))
    print(len(scores))
    print(pts)
    print(scores)

    h_pt_nn_h   = TH1D("h_pt_nn_histo{}".format(tagger.name), "h_pt_nn_histo{}".format(tagger.name), len(pts), 0., 3000)
    #print(pts)
    #print(scores)

    a2h(scores,h_pt_nn_h)

    def score_cut(pt):
        return h_pt_nn_h.GetBinContent(h_pt_nn_h.FindBin(pt))
    tagger.scores["tag_cut"] = np.vectorize(score_cut)(tagger.scores.fjet_pt)
    tagger.signal = tagger.scores[tagger.scores.EventInfo_mcChannelNumber>370000]
    tagger.bg = tagger.scores[tagger.scores.EventInfo_mcChannelNumber<370000]

    tagger.bg_tagged = tagger.bg[tagger.bg.fjet_nnscore > tagger.bg.tag_cut]
    tagger.bg_untagged = tagger.bg[tagger.bg.fjet_nnscore < tagger.bg.tag_cut]
    tagger.signal_tagged = tagger.signal[tagger.signal.fjet_nnscore > tagger.signal.tag_cut]


def get_flat_weight(pt,dsid):

    inFile = TFile.Open( "flat_weights.root" ," READ ")
    flat_bg = inFile.Get("bg_inv")
    flat_sig = inFile.Get("h_sig_inv")

    if dsid > 370000:
        return flat_sig.GetBinContent(flat_sig.FindBin(pt))
    else:
        return flat_bg.GetBinContent(flat_bg.FindBin(pt))

class tagger_scores():
    def __init__(self, name, score_file,working_point):
        intreename = "FlatSubstructureJetTree"
        self.name = name
        self.score_file = score_file

        f = TFile.Open(score_file, 'READ')
        tree = f.Get(intreename)

        #self.events = uproot.open(score_file+":"+intreename)
        #self.scores = self.events.arrays( library="pd")
        branches = []
        mycopy = tree.GetListOfBranches()
        for i in mycopy:
            branches.append(i.GetName())
        print(str(branches))
        arr = tree2array(tree, branches = branches, include_weight = False)
        self.scores = pd.DataFrame(arr)

        print(self.scores.head())
        print(self.scores["EventInfo_mcChannelNumber"].values)

        self.scores["no_weight"]   = np.ones_like(self.scores.fjet_pt.values)
        print("I am here in utils")
        try:
            self.scores["chris_weight"] = (self.scores["fjet_weight_pt_dR"].values)
        except:
            self.scores["chris_weight"] = (self.scores["fjet_weight_pt"].values)


        ## Make the pt spectrum smooth (Chris Delitzsch advice)
        alpha = self.scores[self.scores.EventInfo_mcChannelNumber == 364702][self.scores.fjet_pt > 1000]
        self.scores  = self.scores[self.scores.index.isin(alpha.index) == False]
        alpha = self.scores[self.scores.EventInfo_mcChannelNumber == 364703][self.scores.fjet_pt > 1000]
        self.scores  = self.scores[self.scores.index.isin(alpha.index) == False]
        alpha = self.scores[self.scores.EventInfo_mcChannelNumber == 364704][self.scores.fjet_pt > 2000]
        self.scores  = self.scores[self.scores.index.isin(alpha.index) == False]

        self.signal        = self.scores[self.scores.EventInfo_mcChannelNumber>370000]
        self.signal_tagged = self.signal[self.signal.fjet_nnscore > working_point]

        self.bg          = self.scores[self.scores.EventInfo_mcChannelNumber<370000]
        self.bg_tagged   = self.bg[self.bg.fjet_nnscore > working_point]
        self.bg_untagged = self.bg[self.bg.fjet_nnscore < working_point]

        print (self.name)
        print(self.signal)
        print ("signal ratio:",len(self.signal_tagged.values)/len(self.signal.values))
        print ("bg ratio:",        len(self.bg_tagged.values)/len(self.bg.values))

        print(self.signal["fjet_nnscore"].values)

        self.h_signal = TH1D( "signal{}".format(self.name), "signal{}".format(self.name), 500, 0, 1)
        self.h_bg     = TH1D(     "bg{}".format(self.name),     "bg{}".format(self.name), 500, 0, 1)

        self.h_signal.SetDirectory(0)
        self.h_bg.SetDirectory(0)

        fh(self.h_signal, self.signal["fjet_nnscore"].values, self.signal["chris_weight"].values)
        fh(self.h_bg,         self.bg["fjet_nnscore"].values,     self.bg["chris_weight"].values)

    def get_roc(self):
        tprs, fprs, auc, tpr_wp, fpr_wp = roc_from_histos(self.h_signal, self.h_bg, 0.5)
        return tprs, fprs, auc



def getANNROCresults(axis, NN):
    zNN_x = np.array([5.359762e-11, 0.009999154, 0.01999116, 0.03003638, 0.03992488, 0.05008107, 0.05996809, 0.06998417, 0.07975059, 0.08996194, 0.1000304, 0.1100082, 0.1199365, 0.1299003, 0.1399401, 0.1499697, 0.1600161, 0.1699898, 0.1799898, 0.1899822, 0.2000006, 0.2099966, 0.2197999, 0.2298461, 0.2399201, 0.250004, 0.259998, 0.2700322, 0.279973, 0.2899398, 0.2999996, 0.3100064, 0.3200913, 0.3300106, 0.3398792, 0.3499664, 0.3601058, 0.3699586, 0.3800236, 0.3901443, 0.4000943, 0.4099773, 0.419938, 0.4300162, 0.4400082, 0.4500711, 0.4599526, 0.4699575, 0.4800948, 0.49003, 0.5000187, 0.5099099, 0.5198858, 0.530054, 0.5399972, 0.5499638, 0.5599786, 0.5699802, 0.5799889, 0.5898839, 0.5999322, 0.6099754, 0.6201509, 0.6301104, 0.6400697, 0.6499818, 0.6600177, 0.6699393, 0.6799816, 0.6900021, 0.700002, 0.7100613, 0.7197974, 0.7300616, 0.7400097, 0.7499867, 0.7599711, 0.7701273, 0.780002, 0.7900227, 0.8000471, 0.80999, 0.8199878, 0.8300174, 0.8400352, 0.8500031, 0.8599743, 0.8697525, 0.8799463, 0.889941, 0.9000133, 0.909852, 0.9199605, 0.9301226, 0.9400463, 0.949979, 0.9600671, 0.9700676, 0.9799932, 0.9899974, 1])
    zNN_y = np.array([2.867778e+12, 5930.234, 3665.031, 2455.226, 1586.232, 1279.046, 1121.959, 926.2596, 709.0808, 613.5191, 545.6725, 493.54, 420.899, 384.3847, 349.8579, 312.4534, 287.8106, 259.2606, 245.8432, 226.1111, 211.3946, 195.3024, 182.9064, 170.1389, 153.2901, 143.6002, 133.6891, 124.3174, 116.4907, 107.9008, 103.1832, 96.81649, 91.73322, 86.78455, 81.89352, 75.37916, 71.34418, 67.40619, 63.65544, 61.18966, 57.93535, 54.78706, 51.93011, 49.35323, 46.8307, 44.32984, 42.25477, 40.2392, 38.34341, 36.56929, 34.62039, 32.97139, 31.63376, 30.33788, 29.18099, 27.63851, 26.29479, 24.84247, 23.56773, 22.22711, 21.23503, 20.10873, 19.28947, 18.31277, 17.51248, 16.69247, 15.94066, 15.04993, 14.34487, 13.64683, 13.14976, 12.59437, 11.98574, 11.28208, 10.85309, 10.31724, 9.834648, 9.408443, 9.069162, 8.590538, 8.047843, 7.684222, 7.221932, 6.729962, 6.340407, 5.942391, 5.569178, 5.197152, 4.902587, 4.588132, 4.338056, 4.064786, 3.761534, 3.450641, 3.20575, 2.922286, 2.629, 2.320033, 1.999058, 1.717125, 1.003188])
    zANN_lambda10_x = np.array([1.131714e-11, 0.009918825, 0.02007162, 0.02998319, 0.03997175, 0.05008127, 0.06003026, 0.06988965, 0.07998845, 0.08998601, 0.09999256, 0.1100433, 0.1199873, 0.129989, 0.1399396, 0.1499936, 0.1599913, 0.1700077, 0.1800816, 0.1901243, 0.1998875, 0.2101156, 0.21999, 0.229943, 0.2398725, 0.2499571, 0.2602079, 0.2700333, 0.280135, 0.2899956, 0.2999942, 0.3101231, 0.3198935, 0.329976, 0.3399595, 0.3498552, 0.3600172, 0.3700557, 0.3799915, 0.3899906, 0.4000443, 0.4099927, 0.4199441, 0.4300002, 0.4400033, 0.4499736, 0.4600005, 0.4702282, 0.480123, 0.4899349, 0.5000647, 0.5100911, 0.5198551, 0.5298495, 0.539887, 0.5502125, 0.5600565, 0.5701899, 0.5800196, 0.5899909, 0.6000111, 0.6099331, 0.6198772, 0.6301648, 0.6398515, 0.6500098, 0.6600693, 0.6700729, 0.6801797, 0.6900375, 0.6999045, 0.7099951, 0.7200945, 0.7301415, 0.7399759, 0.7500459, 0.7599822, 0.7700194, 0.7800044, 0.7898643, 0.8000109, 0.8100498, 0.82002, 0.8299957, 0.8399849, 0.8499171, 0.8598864, 0.8699981, 0.879894, 0.89001, 0.9000118, 0.9100013, 0.9199181, 0.9299524, 0.9400149, 0.9500282, 0.9598884, 0.9700237, 0.9799, 0.9900077, 1])
    zANN_lambda10_y = np.array([5.056011e+09, 1187.239, 660.5933, 497.4458, 395.1048, 326.8497, 269.7756, 229.3882, 191.8182, 169.5777, 149.2982, 131.5672, 119.2591, 107.2369, 95.57749, 87.72122, 80.28384, 73.94049, 67.24739, 62.2697, 56.86721, 53.9289, 50.23599, 47.49033, 45.43967, 42.5143, 40.20649, 38.01617, 35.23764, 33.16732, 31.24131, 29.55421, 28.03851, 26.84014, 25.74013, 24.50817, 23.33966, 21.98521, 20.71675, 19.85033, 18.7976, 18.03967, 17.41941, 16.48102, 15.88681, 15.17202, 14.53261, 13.87051, 13.50393, 12.87075, 12.4527, 12.07215, 11.56056, 11.10838, 10.68442, 10.22038, 9.748485, 9.361632, 8.927393, 8.576681, 8.243342, 7.871607, 7.512402, 7.19025, 6.917642, 6.625158, 6.416827, 6.157932, 5.939743, 5.689208, 5.464191, 5.25604, 5.123059, 4.918376, 4.729624, 4.568797, 4.399293, 4.229056, 4.054728, 3.93252, 3.7363, 3.616771, 3.489878, 3.330711, 3.181462, 3.04097, 2.935096, 2.800061, 2.644747, 2.50727, 2.387536, 2.271244, 2.15915, 2.037161, 1.930465, 1.824921, 1.703094, 1.566049, 1.426781, 1.259529, 1.000001])
    if(axis=="x" and NN=="NN"): return zNN_x
    if(axis=="y" and NN=="NN"): return zNN_y
    if(axis=="x" and NN=="ANN"): return zANN_lambda10_x
    if(axis=="y" and NN=="ANN"): return zANN_lambda10_y



class trivar_scores():
    def __init__(self, name, score_file):
        intreename = "FlatSubstructureJetTree"
        self.name = name
        self.score_file = score_file
        #self.events = uproot.open(score_file+":"+intreename)
        #self.scores = self.events.arrays( library="pd")
        branches = []
        mycopy = tree.GetListOfBranches()
        for i in mycopy:
            branches.append(i)
        arr = tree2array(tree, branches = branches, include_weight = False)
        self.scores = pd.DataFrame(arr)

        self.scores["chris_weight"] = (self.scores["fjet_weight_pt"])
        self.scores["xsec_weight"] = np.vectorize(assign_weights)(self.scores["EventInfo_mcChannelNumber"],self.scores["EventInfo_mcEventWeight"])
        self.scores["flat_weight"] = np.vectorize(get_flat_weight)(self.scores["fjet_pt"],self.scores["EventInfo_mcChannelNumber"])
        self.scores["no_weight"] = np.ones_like(self.scores.fjet_pt)

        ## Make the pt spectrum smooth (Chris Delitzsch advice)
        alpha = self.scores[self.scores.EventInfo_mcChannelNumber == 364702][self.scores.fjet_pt > 1000]
        self.scores  = self.scores[self.scores.index.isin(alpha.index) == False]
        alpha = self.scores[self.scores.EventInfo_mcChannelNumber == 364703][self.scores.fjet_pt > 1000]
        self.scores  = self.scores[self.scores.index.isin(alpha.index) == False]
        alpha = self.scores[self.scores.EventInfo_mcChannelNumber == 364704][self.scores.fjet_pt > 2000]
        self.scores  = self.scores[self.scores.index.isin(alpha.index) == False]

        #coeffs_mass_high = [143.346574141,-0.226450777605,0.000389338881315,-3.3948387014e-07,1.6059552279e-10,-3.89697376333e-14,3.81538674411e-18]
        #coeffs_mass_low = [78.0015279678,-0.0607637891015,0.000154878939873,-1.85055756284e-07,1.06053761725e-10,-2.9181422716e-14,3.09607176224e-18]
        #coeffs_d2 = [1.86287598712,-0.00286891844597,6.51440728353e-06,-7.14076683933e-09,3.97453495445e-12,-1.07885298604e-15,1.1338084323e-19]
        #coeffs_ntrk = [18.1029210508,0.0328710277742,-4.90091461191e-05,3.72086065666e-08,-1.57111307275e-11,3.50912856537e-15,-3.2345326821e-19]


        coeffs_mass_low = [77.85195198272105,-0.04190870755297197,0.00010148243081053968,-1.2646715469383716e-07,7.579631867406234e-11,-2.1810858771189926e-14,2.4131259557938418e-18]
        coeffs_mass_high = [138.40389824173184,-0.1841270515643543,0.0003150778420142889,-2.8146937922756945e-07,1.3687749824011263e-10,-3.370270044494874e-14,3.2886002834089895e-18]
        coeffs_d2 = [1.1962224520689877,0.0007051153225402016,-7.368355018553183e-07,-5.841704226982689e-11,4.1301607038564777e-13,-1.933293321407319e-16,2.7326862198181657e-20]
        coeffs_ntrk = [15.838972910273808,0.059376592913538105,-0.00010408419300237432,9.238395877087256e-08,-4.458514804353202e-11,1.1054941188725808e-14,-1.1013796203558003e-18]

        self.scores["d2_cut"] = np.vectorize(lambda x:coeffs_d2[0]+x*coeffs_d2[1]+coeffs_d2[2]*x**2+coeffs_d2[3]*x**3+coeffs_d2[4]*x**4+coeffs_d2[5]*x**5+coeffs_d2[6]*x**6)(self.scores.fjet_pt)
        self.scores["ntrk_cut"] = np.vectorize(lambda x:coeffs_ntrk[0]+x*coeffs_ntrk[1]+coeffs_ntrk[2]*x**2+coeffs_ntrk[3]*x**3+coeffs_ntrk[4]*x**4+coeffs_ntrk[5]*x**5+coeffs_ntrk[6]*x**6)(self.scores.fjet_pt)
        self.scores["mlow_cut"] = np.vectorize(lambda x:coeffs_mass_low[0]+x*coeffs_mass_low[1]+coeffs_mass_low[2]*x**2+coeffs_mass_low[3]*x**3+coeffs_mass_low[4]*x**4+coeffs_mass_low[5]*x**5+coeffs_mass_low[6]*x**6)(self.scores.fjet_pt)
        self.scores["mhigh_cut"] = np.vectorize(lambda x:coeffs_mass_high[0]+x*coeffs_mass_high[1]+coeffs_mass_high[2]*x**2+coeffs_mass_high[3]*x**3+coeffs_mass_high[4]*x**4+coeffs_mass_high[5]*x**5+coeffs_mass_high[6]*x**6)(self.scores.fjet_pt)

        self.signal = self.scores[self.scores.EventInfo_mcChannelNumber>370000]
        self.bg = self.scores[self.scores.EventInfo_mcChannelNumber<370000]

        self.signal_tagged = self.signal[self.signal.fjet_m > self.signal["mlow_cut"]][self.signal.fjet_m < self.signal["mhigh_cut"]][self.signal.fjet_d2 < self.signal["d2_cut"]][self.signal.fjet_ntrk < self.signal["ntrk_cut"]]
        self.bg_tagged = self.bg[self.bg.fjet_m > self.bg["mlow_cut"]][self.bg.fjet_m < self.bg["mhigh_cut"]][self.bg.fjet_d2 < self.bg["d2_cut"]][self.bg.fjet_ntrk < self.bg["ntrk_cut"]]
        self.bg_untagged  = self.bg[self.bg.index.isin(self.bg_tagged.index) == False]

        print (self.name)
        print ("signal ratio:" ,len(self.signal_tagged)/len(self.signal))
        print ("bg ratio:" ,len(self.bg_tagged)/len(self.bg))

def JSD (P, Q, base=2):
    """Compute Jensen-Shannon divergence (JSD) of two distribtions.
    From: [https://stackoverflow.com/a/27432724]

    Arguments:
        P: First distribution of variable as a numpy array.
        Q: Second distribution of variable as a numpy array.
        base: Logarithmic base to use when computing KL-divergence.

    Returns:
        Jensen-Shannon divergence of `P` and `Q`.
    """
    p = P / np.sum(P)
    q = Q / np.sum(Q)
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m, base=base) + entropy(q, m, base=base))


def scores_separation(tagger):

    c1 = ap.canvas(num_pads=1, batch=True)

    bins = np.linspace(0, 1, 100+1)
    colours = ["#f7931e", "#07407b"]
    hSignal = c1.hist(np.asarray(tagger.signal["fjet_nnscore"]), bins=bins, fillcolor=TColor.GetColor(colours[0]), alpha=0.35, label="Signal", option="HIST")
    hBkg    = c1.hist(np.asarray(tagger.bg["fjet_nnscore"]),     bins=bins, fillcolor=TColor.GetColor(colours[1]), alpha=0.35, label="Background", option="HIST")

    c1.xlabel('NN score')
    c1.ylabel('')

    c1.text(["#sqrt{s} = 13 TeV, #it{W} tagging",
            "anti-k_{t} R=1.0 UFO jets",
            "Tagger = {}".format(tagger.name),
            ], qualifier='Simulation Internal')
    c1.log()
    c1.legend()
    c1.save("/eos/user/t/tmlinare/Lund_tagger/ljptagger/Models/Graphs/classification_{}.png".format(tagger.name))

def scores_separation_pt(tagger, minpt, maxpt):

    c1 = ap.canvas(num_pads=1, batch=True)

    bins = np.linspace(0, 1, 100+1)
    colours = ["#f7931e", "#07407b"]
    signal_pt = tagger.signal[tagger.signal.fjet_pt < maxpt ][tagger.signal.fjet_pt > minpt ]
    bg_pt     = tagger.bg[tagger.bg.fjet_pt < maxpt ][tagger.bg.fjet_pt > minpt ]

    hSignal = c1.hist(np.asarray(signal_pt["fjet_nnscore"]), bins=bins, fillcolor=TColor.GetColor(colours[0]), alpha=0.35, label="Signal", option="HIST")
    hBkg    = c1.hist(np.asarray(bg_pt["fjet_nnscore"]),     bins=bins, fillcolor=TColor.GetColor(colours[1]), alpha=0.35, label="Background", option="HIST")

    c1.xlabel('NN score')
    c1.ylabel('')

    c1.text(["#sqrt{s} = 13 TeV, #it{W} tagging",
            "anti-k_{t} R=1.0 UFO jets",
            "Tagger = {}".format(tagger.name),
            "p_{T} #in [%d, %d] GeV" % (minpt, maxpt),
            ], qualifier='Simulation Internal')
    c1.log()
    c1.legend()
    c1.save("/eos/user/t/tmlinare/Lund_tagger/ljptagger/Models/Graphs/classification_{}_pt_{}_{}.png".format(tagger.name,minpt,maxpt))

def make_efficiencies_pt(taggers, minpt, maxpt,weight="chris_weight"):
    #plt.figure(figsize=(16,12))

    colours = [ROOT.kViolet + 7, ROOT.kAzure + 7, ROOT.kTeal, ROOT.kSpring - 2, ROOT.kOrange - 3, ROOT.kPink,  ROOT.kPink+3]

    count = 0
    c1 = ap.canvas(num_pads=1, batch=True)
    for t in taggers:

        if taggers[t].name == "3var":
            bg_tag = taggers[t].bg_tagged[taggers[t].bg.fjet_pt < maxpt ][taggers[t].bg.fjet_pt > minpt ]
            bg_all = taggers[t].bg[taggers[t].bg.fjet_pt < maxpt ][taggers[t].bg.fjet_pt > minpt ]
            signal_tag = taggers[t].signal_tagged[taggers[t].signal.fjet_pt < maxpt ][taggers[t].signal.fjet_pt > minpt ]
            signal = taggers[t].signal[taggers[t].signal.fjet_pt < maxpt ][taggers[t].signal.fjet_pt > minpt ]

            sig_eff = (signal_tag.chris_weight.sum())/signal.chris_weight.sum()
            bg_eff = (bg_all.chris_weight.sum())/(bg_tag.chris_weight.sum())

            ## Fill 3-var tagger for [minpt, maxpt] range
            h = TGraph()
            h.SetPoint(0, sig_eff, bg_eff)
            c1.graph(h, linecolor=ROOT.kWhite, markercolor=ROOT.kRed, markerstyle=20, option="P", label="3-var tagger")


        else:
            bg_all = taggers[t].bg[taggers[t].bg.fjet_pt < maxpt ][taggers[t].bg.fjet_pt > minpt ]
            signal = taggers[t].signal[taggers[t].signal.fjet_pt < maxpt ][taggers[t].signal.fjet_pt > minpt ]
            h_signal = TH1D ("signal{}".format(taggers[t].name),"signal{}".format(taggers[t].name),500,0,1)
            h_bg = TH1D ("bg{}".format(taggers[t].name),"bg{}".format(taggers[t].name),500,0,1)
            fh(h_signal,signal["fjet_nnscore"],signal[weight])
            fh(h_bg,bg_all["fjet_nnscore"],bg_all[weight])
            tprs, fprs, auc, tpr_wp, fpr_wp = roc_from_histos(h_signal, h_bg,0.5)
            ## protect for inf
            inv = 1/fprs
            inv[inv == np.inf] = 1e10
            h = TGraph(len(tprs), tprs, inv)
            c1.graph(h, linestyle=1, linecolor=colours[count], markercolor=colours[count], markerstyle=1, option="AL", label=taggers[t].name)
            ## Fill random guessing
            h = TGraph(len(tprs), np.linspace(0.0001, 1, len(tprs)), 1/np.linspace(0.0001, 1, len(tprs)))
            c1.graph(h, linestyle=2, linecolor=ROOT.kBlack, option="L", label="Random guessing")

        count+=1

    c1.xlabel('Signal efficiency')
    c1.ylabel('Background rejection')
    c1.xlim(0.1, 1) ## c1.xlim(0.2, 1)
    c1.ylim(1, 1e7)

    c1.text(["#sqrt{s} = 13 TeV, #it{W} tagging",
            "anti-k_{t} R=1.0 UFO jets",
            "p_{T} #in [%d, %d] GeV" % (minpt, maxpt),
            ], qualifier='Simulation Internal')
    c1.log()
    c1.legend()
    c1.save("/eos/user/t/tmlinare/Lund_tagger/ljptagger/Models/Graphs/Efficiencies_3var_taggers_{}_{}.png".format(minpt, maxpt))


def mass_sculpting(tagger,weight="chris_weight"):

    bins = np.linspace(50, 300, (300 - 50) // 5 + 1, endpoint=True)

    print(tagger.name)

    p, _ = np.histogram(tagger.bg_untagged["fjet_m"].values, bins=bins, density=1.)
    f, _ = np.histogram(tagger.bg_tagged["fjet_m"].values, bins=bins, density=1.)
    jsd = JSD(p,f)


    c1 = ap.canvas(num_pads=1, batch=True)
    p0= c1.pads()

    #hTotalBG  = c1.hist(np.asarray(tagger.bg["fjet_m"]), bins=bins, weights=np.asarray(tagger.bg[weight]), fillstyle = 3353, fillcolor=ROOT.KGray + 2, label="Total BG", linestyle=1, option='HIST', normalise=True)
    hUntagBG  = c1.hist(np.asarray(tagger.bg_untagged["fjet_m"]), bins=bins, weights=np.asarray(tagger.bg_untagged[weight]), fillcolor=ROOT.kGray + 2,  label="QCD jets", option='HIST', normalise=True) ## Untagged BG
    hSignal   = c1.hist(np.asarray(tagger.signal["fjet_m"]), bins=bins, weights=np.asarray(tagger.signal[weight]), fillstyle = 3353, fillcolor=ROOT.kGray+1, label="Signal", linestyle=1, linecolor=ROOT.kGray+1, option='HIST', normalise=True)
    hTaggedBG = c1.hist(np.asarray(tagger.bg_tagged["fjet_m"]), bins=bins, weights=np.asarray(tagger.bg_tagged[weight]), label="QCD jets after ANN", linecolor=9, linestyle=2, option='HIST', normalise=True)  ## Tagged BG


    c1.xlabel('Large-R jet mass [GeV]')
    c1.ylabel('Fraction of jets / 5 GeV')

    c1.text(["#sqrt{s} = 13 TeV, #it{W} tagging",
            "anti-k_{t} R=1.0 UFO jets",
            "JSD = {}".format(round(jsd,5)),
            ], qualifier='Simulation Internal')
    c1.log()
    c1.legend()
    c1.save("/eos/user/t/tmlinare/Lund_tagger/ljptagger/Models/Graphs/mass_sculpting.png")


def mass_sculpting_ptcut(tagger,minpt,maxpt,weight="chris_weight"):

    bins = np.linspace(50, 300, (300 - 50) // 5 + 1, endpoint=True)

    #bg_all = tagger.bg[tagger.bg.fjet_pt < maxpt ][tagger.bg.fjet_pt > minpt ]
    signal = tagger.signal[tagger.signal.fjet_pt < maxpt ][tagger.signal.fjet_pt > minpt ]
    bgu = tagger.bg_untagged[tagger.bg_untagged.fjet_pt < maxpt ][tagger.bg_untagged.fjet_pt > minpt ]
    bgt = tagger.bg_tagged[tagger.bg_tagged.fjet_pt < maxpt ][tagger.bg_tagged.fjet_pt > minpt ]

    p, _ = np.histogram(bgu, bins=bins, density=1.)
    f, _ = np.histogram(bgt, bins=bins, density=1.)
    jsd = JSD(p,f)

    c1 = ap.canvas(num_pads=1, batch=True)
    p0= c1.pads()

    hUntagBG  = c1.hist(np.asarray(bgu["fjet_m"]), bins=bins, weights=np.asarray(bgu[weight]), fillcolor=ROOT.kGray + 2,  label="QCD jets", option='HIST', normalise=True) ## Untagged BG
    hSignal   = c1.hist(np.asarray(signal["fjet_m"]), bins=bins, weights=np.asarray(signal[weight]), fillstyle = 3353, fillcolor=ROOT.kGray+1, label="Signal", linestyle=1, linecolor=ROOT.kGray+1, option='HIST', normalise=True)
    hTaggedBG = c1.hist(np.asarray(bgt["fjet_m"]), bins=bins, weights=np.asarray(bgt[weight]), label="QCD jets after ANN", linecolor=9, linestyle=2, option='HIST', normalise=True)  ## Tagged BG


    c1.xlabel('Large-R jet mass [GeV]')
    c1.ylabel('Fraction of jets / 5 GeV')

    c1.text(["#sqrt{s} = 13 TeV, #it{W} tagging",
            "anti-k_{t} R=1.0 UFO jets",
            "p_{T} #in [%d, %d] GeV" % (minpt, maxpt),
            "JSD = {}".format(round(jsd,5)),
            ], qualifier='Simulation Internal')
    c1.log()
    c1.legend()
    #c1.ylim(0, hTotalBG.GetMaximum())
    c1.save("/eos/user/t/tmlinare/Lund_tagger/ljptagger/Models/Graphs/mass_sculpting_{}_{}.png".format(minpt, maxpt))

def mass_bgrej(taggers,weight="chris_weight"):
    plt.figure(figsize=[16,12])
    nbins = 360


    h_bg_total = TH1D ("bg_total","bgtotal",    nbins,0,3000)
    #fh(h_bg_total,taggers["3var"].bg["fjet_m"],taggers["3var"].bg[weight])
    a_bkg = h2a(h_bg_total)

    for t in taggers:
        h_bg = TH1D ("bg_{}".format(taggers[t].name),"bg_{}".format(taggers[t].name),nbins,0,3000)
        fh(h_bg,taggers[t].bg_tagged["fjet_m"],taggers[t].bg_tagged[weight])
        bg = h2a(h_bg)
        rat = a_bkg/bg
        rat = rat[np.logical_not(np.isnan(rat))]


        plt.semilogy(np.linspace(0, 3000, nbins),a_bkg/bg,
                     label="1/BG rejection {}".format(taggers[t].name))

    plt.ylabel("Reweighted event counts")
    plt.xlabel("Jet M $[GeV]$")


    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    #plt.xlim(5,500)
    #plt.ylim(10**(-16), 10**(-0))
    plt.legend()
    plt.legend(prop={'size': 15})
    plt.savefig("/eos/user/t/tmlinare/Lund_tagger/ljptagger/Models/Graphs/mass_bgrej.png",dpi=500)

    #plt.gca().invert_yaxis()
    plt.show()

def mass_sigeff(taggers,weight="chris_weight"):
    plt.figure(figsize=[16,12])
    nbins = 550
    for t in taggers:
        some_tagger = t

    h_sig_total = TH1D ("sig_total","sigtotal",nbins,0,3000)
#    MASSBINS = np.linspace(200, 3000, (300 - 40) // 5 + 1, endpoint=True)

    fh(h_sig_total,taggers[some_tagger].signal["fjet_m"],taggers[some_tagger].signal[weight])
    a_sig = h2a(h_sig_total)
#    plt.semilogy(np.linspace(0, 3000, 100), a_sig, label="all signal")

    for t in taggers:
        h_sig = TH1D ("sig_{}".format(taggers[t].name),"sig_{}".format(taggers[t].name),nbins,0,3000)
        fh(h_sig,taggers[t].signal_tagged["fjet_m"],taggers[t].signal_tagged[weight])
        sig = h2a(h_sig)
        plt.plot(np.linspace(0, 3000, nbins),sig/a_sig,
                     label="tagged signal {}".format(taggers[t].name))
        rat = sig/a_sig
        rat = rat[np.logical_not(np.isnan(rat))]
        print ("sig eff mean:", (rat).mean())

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.ylabel("Reweighted event counts")
    plt.xlabel("Jet pT $[GeV]$")
    #plt.xlim(5,500)
    plt.ylim(10**(-16), 10**(-0))
    plt.legend()
    plt.legend(prop={'size': 15})
    plt.savefig("/eos/user/t/tmlinare/Lund_tagger/ljptagger/Models/Graphs/mass_sigeff.png",dpi=500)
    #plt.gca().invert_yaxis()
    plt.show()



def pt_spectrum(taggers,weight="chris_weight"):



    colours = [ROOT.kViolet + 7, ROOT.kAzure + 7, ROOT.kTeal, ROOT.kSpring - 2, ROOT.kOrange - 3, ROOT.kPink,  ROOT.kPink+3]
    c = ap.canvas(num_pads=1, batch=True)
    p0= c.pads()

    nbins = 50
    h_bg_total = TH1D ("bg_total","bg_total",nbins,0,3000)
    fh(h_bg_total,taggers["LundNet_class"].bg["fjet_pt"],taggers["LundNet_class"].bg[weight])
    h_bg_total = c.hist(h_bg_total, linecolor=colours[0], linestyle=2, option='HIST', label="All background")
    #fh(h_bg_total,taggers["LundNet_class"].bg["fjet_pt"])
    a_bkg = h2a(h_bg_total)



    h_bg_plots = []
    bg_plots = []
    count = 1
    for t in taggers:
        bg_plots.append (TH1D ("bg_{}".format(taggers[t].name),"bg_{}".format(taggers[t].name),nbins,0,3000))
        fh(bg_plots[-1],taggers[t].bg_tagged["fjet_pt"],taggers[t].bg_tagged[weight])
        h = c.hist(bg_plots[-1], linecolor=colours[count], linestyle=2, option='HIST', label=taggers[t].name)
        #fh(bg_plots[-1],taggers[t].bg_tagged["fjet_pt"])
        bg_plots.append( h2a(bg_plots[-1]))
        count+=1
        #plt.semilogy(np.linspace(0, 3000, nbins),bg_plots[-1],
        #             label="tagged background {}".format(taggers[t].name))
    #plt.semilogy(np.linspace(0, 3000, nbins), a_bkg, label="all background")


    c.text(["#sqrt{s} = 13 TeV, #it{W} tagging",] , qualifier='Simulation Internal')

    # -- Axis labels
    c.xlabel('Large-R jet p_{T} [GeV]')
    c.ylabel('Events')

    # -- Log
    c.log()
    c.legend()
    c.save("/eos/user/t/tmlinare/Lund_tagger/ljptagger/Models/Graphs/ptspectrum.png")


def pt_bgrej(taggers,weight="chris_weight"):

    colours = [ROOT.kViolet + 7, ROOT.kAzure + 7, ROOT.kTeal, ROOT.kSpring - 2, ROOT.kOrange - 3, ROOT.kPink,  ROOT.kPink+3]

    bins = np.linspace(200, 3000, 16 + 1, endpoint=True)
    c = ap.canvas(num_pads=1, batch=True)
    count = 0
    for t in taggers:

        ## Get total background
        h_bg_total = c.hist(np.array(taggers[t].bg["fjet_pt"]), weights=np.array(taggers[t].bg[weight]), bins=bins, display=False)
        ## Get tagged background
        h_bg = c.hist(np.array(taggers[t].bg_tagged["fjet_pt"]), weights=np.array(taggers[t].bg_tagged[weight]), bins=bins, display=False)

        ## Calculate bkg rejection (1/epsilon bkg = total bkg / tagged bkg)
        c.ratio_plot((h_bg_total,      h_bg), option='HIST', offset=1, bins=bins, label=taggers[t].name, linecolor=colours[count])

        count +=1

    c.xlabel('Large-#it{R} jet p_{T} [GeV]')
    c.ylabel('Background rejection 1/#epsilon^{rel}_{bkg}')

    c.text(["#sqrt{s} = 13 TeV, #it{W} tagging",
            "anti-k_{t} R=1.0 UFO jets",
            ], qualifier='Simulation Internal')
    c.log()
    c.legend()
    c.ylim(1e1, 1e5)
    c.save("/eos/user/t/tmlinare/Lund_tagger/ljptagger/Models/Graphs/pt_bgrej.png")



def pt_sigeff(taggers,weight="chris_weight"):
    plt.figure(figsize=[16,12])
    nbins = 50
    for t in taggers:
        some_tagger = t

    h_sig_total = TH1D ("sig_total","sigtotal",nbins,0,3000)
#    MASSBINS = np.linspace(200, 3000, (300 - 40) // 5 + 1, endpoint=True)

    fh(h_sig_total,taggers[some_tagger].signal["fjet_pt"],taggers[some_tagger].signal[weight])
    a_sig = h2a(h_sig_total)
#    plt.semilogy(np.linspace(0, 3000, 100), a_sig, label="all signal")

    for t in taggers:
        h_sig = TH1D ("sig_{}".format(taggers[t].name),"sig_{}".format(taggers[t].name),nbins,0,3000)
        fh(h_sig,taggers[t].signal_tagged["fjet_pt"],taggers[t].signal_tagged[weight])
        sig = h2a(h_sig)
        plt.plot(np.linspace(0, 3000, nbins),sig/a_sig,
                     label="tagged signal {}".format(taggers[t].name))
        rat = sig/a_sig
        rat = rat[np.logical_not(np.isnan(rat))]
        print ("sig eff mean:", (rat).mean())


    plt.ylabel("Reweighted event counts")
    plt.xlabel("Jet pT $[GeV]$")
    #plt.xlim(5,500)
    plt.ylim(10**(-16), 10**(-0))
    plt.legend()
    plt.legend(prop={'size': 15})

    #plt.gca().invert_yaxis()
    plt.show()



def weights(tagger):
    plt.figure(figsize=[16,8])
    kwargs = dict(alpha = 0.75, bins = 50, density = True, stacked = True,range=(0,1))
    plt.title(label=tagger.name, fontdict=None, loc='center', pad=None)

    #plt.hist(tagger.signal["fjet_weight_pt_dR"],  ** kwargs, label="Signal")
    #plt.hist(tagger.bg["fjet_weight_pt_dR"], ** kwargs, label="Background")

    plt.hist(tagger.signal["fjet_weight_pt_dR"], label="Signal")
    plt.hist(tagger.bg["fjet_weight_pt_dR"], label="Background")
    plt.legend(fontsize=30)
    plt.show()
