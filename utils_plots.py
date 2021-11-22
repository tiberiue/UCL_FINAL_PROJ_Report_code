import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import ROOT as root
import math

from root_numpy import fill_hist  as fh
from root_numpy import array2hist as a2h
from root_numpy import hist2array as h2a



def roc_from_histos(h_sig, h_bkg, wpcut=None):
    """
    Compute a ROC curve given two histograms.
    """
    
    tprs = []
    fprs = []
    
    num_bins = h_sig.GetXaxis().GetNbins()
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
    plt.figure(figsize=(16,12))

    for t in taggers:
        tprs, fprs, auc = taggers[t].get_roc()
        plt.plot(fprs, tprs, label="{0}, AUC = {1:.3f}".format(taggers[t].name, auc))

    
    plt.title("NN taggers ROC curve")
    plt.xlabel("$\epsilon_{bkg}$")
    plt.ylabel("$\epsilon_{sig}$")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.legend()
    #plt.savefig("top_roc_fulld_new",dpi=500)
    plt.show()



def make_efficiencies(taggers):
    plt.figure(figsize=(16,12))

    for t in taggers:
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
    plt.savefig("w_cor_uncor1",dpi=500)
    plt.show()


def get_eff_score(pt_vs_score,wp):
    scores_projection = pt_vs_score.ProjectionX()
    pt_value = []
    tag_score = []
    for ptbin in range(1,pt_vs_score.GetNbinsX()):
        curcont = 0
        pt_value.append(scores_projection.GetBinCenter(ptbin))
        if scores_projection.GetBinContent(ptbin)==0:
            tag_score.append(0)
            continue
        for scorebin in range(1, pt_vs_score.GetNbinsX()):
            curcont += pt_vs_score.GetBinContent(ptbin, scorebin)
            if curcont/scores_projection.GetBinContent(ptbin) >= wp:
                tag_score.append(scorebin/100)
                break
    return pt_value, tag_score


def wp50_cut(p,pt):
    return p[0]+p[1]/(p[2]+math.exp(p[3]*(pt+p[4])))


def get_wp_tag(tagger, wp):
    h_pt_nn   = root.TH2D( f"h_pt_nn{tagger.name}", f"h_pt_nn{tagger.name}", 100, 0., 3000,100,0,1 )
    for pt,nn in zip(tagger.signal["fjet_pt"],tagger.signal["fjet_nnscore"]):
        h_pt_nn.Fill(pt,nn)        
    pts, scores = get_eff_score(h_pt_nn,wp)
    scores = scores[6:]
    pts = pts [6:]
    gra = root.TGraph(len(pts), np.array(pts).astype("float"), np.array(scores).astype("float"))
    fitfunc = root.TF1("fit", "[p0]+[p1]/([p2]+exp([p3]*(x+[p4])))", 200, 2700) #exponential sigmoid fit (best so far)
    #fitfunc = root.TF1("fit", "pol10", 200, 2700) #12th order polynomial fit 
    gra.Fit(fitfunc,"R,S")
    c = root.TCanvas(f"myCanvasName{tagger.name}",f"The Canvas Title{tagger.name}",800,600)

    gra.Draw()

    p = fitfunc.GetParameters()
    tagger.scores["tag_cut"] = np.vectorize(lambda x:p[0]+p[1]/(p[2]+math.exp(p[3]*(x+p[4]))))(tagger.scores.fjet_pt)
    tagger.signal = tagger.scores[tagger.scores.EventInfo_mcChannelNumber>370000]
    tagger.bg = tagger.scores[tagger.scores.EventInfo_mcChannelNumber<370000]

    tagger.bg_tagged = tagger.bg[tagger.bg.fjet_nnscore > tagger.bg.tag_cut]
    tagger.bg_untagged = tagger.bg[tagger.bg.fjet_nnscore < tagger.bg.tag_cut]
    tagger.signal_tagged = tagger.signal[tagger.signal.fjet_nnscore > tagger.signal.tag_cut]
    c.Draw()

class tagger_scores():

    def __init__(self, name, score_file):
        intreename = "FlatSubstructureJetTree"
        self.name = name
        self.score_file = score_file
        self.events = uproot.open(score_file+":"+intreename)
        self.scores = self.events.arrays( library="pd")
        self.scores["xsec_weight"] = np.vectorize(assign_weights)(self.scores["EventInfo_mcChannelNumber"],self.scores["EventInfo_mcEventWeight"])
#        self.scores["xsec_weight"] = (self.scores["fjet_weight_pt_dR"])

        self.signal = self.scores[self.scores.EventInfo_mcChannelNumber>370000]
        self.signal_tagged = self.signal[self.signal.fjet_nnscore > 0.5]

        self.bg = self.scores[self.scores.EventInfo_mcChannelNumber<370000]
        self.bg_tagged = self.bg[self.bg.fjet_nnscore > 0.5]
        self.bg_untagged = self.bg[self.bg.fjet_nnscore < 0.5]

        print (self.name)
        print ("signal ratio:" ,len(self.signal_tagged)/len(self.signal))
        print ("bg ratio:" ,len(self.bg_tagged)/len(self.bg))
        self.h_signal = root.TH1D (f"signal{self.name}",f"signal{self.name}",500,0,1)
        self.h_bg = root.TH1D (f"bg{self.name}",f"bg{self.name}",500,0,1)
        fh(self.h_signal,self.signal["fjet_nnscore"],self.signal["xsec_weight"])
        fh(self.h_bg,self.bg["fjet_nnscore"],self.bg["xsec_weight"])
    
    def get_roc(self):
        tprs, fprs, auc, tpr_wp, fpr_wp = roc_from_histos(self.h_signal, self.h_bg,0.13)
        return tprs, fprs, auc


def scores_separation(tagger):
    plt.figure(figsize=[16,8])
    kwargs = dict(alpha = 0.75, bins = 50, density = True, stacked = True,range=(0,1))

    plt.hist(tagger.signal["fjet_nnscore"],  ** kwargs, label="Signal")
    plt.hist(tagger.bg["fjet_nnscore"], ** kwargs, label="Background")
    plt.legend(fontsize=30)
    plt.show()

def mass_sculpting(tagger):
    plt.figure(figsize=[16,8])
    kwargs = dict(alpha = 0.9, bins = 100, density = True, stacked = True,range=(50,300))
#    plt.text(100, 0.03, f" {tagger.name}", fontsize=20)    

    plt.hist(tagger.bg["fjet_m"], ** kwargs, weights = tagger.bg["EventInfo_mcEventWeight"], label="Total BG")
    plt.hist(tagger.bg_untagged["fjet_m"], ** kwargs, weights = tagger.bg_untagged["EventInfo_mcEventWeight"],  label="Untagged BG")
    plt.hist(tagger.signal["fjet_m"], ** kwargs, weights = tagger.signal["EventInfo_mcEventWeight"], label="Signal")
    plt.hist(tagger.bg_tagged["fjet_m"], ** kwargs, weights = tagger.bg_tagged["EventInfo_mcEventWeight"],  label="Tagged BG")

    plt.legend(fontsize=30)
    plt.show()
    
def pt_spectrum(taggers):
    plt.figure(figsize=[16,12])

    for t in taggers:
        some_tagger = t

    h_bg_total = root.TH1D (f"bg_total",f"bgtotal",100,0,3000)
#    MASSBINS = np.linspace(200, 3000, (300 - 40) // 5 + 1, endpoint=True)

    fh(h_bg_total,taggers[some_tagger].bg["fjet_pt"],taggers[some_tagger].bg["EventInfo_mcEventWeight"])
    a_bkg = h2a(h_bg_total)
    plt.semilogy(np.linspace(0, 3000, 100), a_bkg, label="all background")
    
    for t in taggers:
        h_bg = root.TH1D (f"bg_{taggers[t].name}",f"bg_{taggers[t].name}",100,0,3000)
        fh(h_bg,taggers[t].bg_tagged["fjet_pt"],taggers[t].bg_tagged["EventInfo_mcEventWeight"])
        bg = h2a(h_bg)
        plt.semilogy(np.linspace(0, 3000, 100),bg, 
                     label="tagged background {}".format(taggers[t].name))

    

    plt.ylabel("Reweighted event counts")
    plt.xlabel("Jet pT $[GeV]$")
    #plt.xlim(5,500)
    #plt.ylim(10**(-16), 10**(-0))
    plt.legend()
    #plt.gca().invert_yaxis()
    plt.show()
    
    
    
def pt_bgrej(taggers):
    plt.figure(figsize=[16,12])
    nbins = 200

    for t in taggers:
        some_tagger = t

    h_bg_total = root.TH1D (f"bg_total",f"bgtotal",    nbins,0,3000)
#    MASSBINS = np.linspace(200, 3000, (300 - 40) // 5 + 1, endpoint=True)

    fh(h_bg_total,taggers[some_tagger].bg["fjet_pt"],taggers[some_tagger].bg["EventInfo_mcEventWeight"]*taggers[some_tagger].bg["xsec_weight"])
    a_bkg = h2a(h_bg_total)
    #plt.semilogy(np.linspace(0, 3000, 100), a_bkg, label="all background")
    
    for t in taggers:
        h_bg = root.TH1D (f"bg_{taggers[t].name}",f"bg_{taggers[t].name}",nbins,0,3000)
        fh(h_bg,taggers[t].bg_tagged["fjet_pt"],taggers[t].bg_tagged["EventInfo_mcEventWeight"]*taggers[t].bg_tagged["xsec_weight"])
        bg = h2a(h_bg)
        plt.semilogy(np.linspace(0, 3000, nbins),a_bkg/bg, 
                     label="1/BG rejection {}".format(taggers[t].name))

    

    plt.ylabel("Reweighted event counts")
    plt.xlabel("Jet pT $[GeV]$")
    #plt.xlim(5,500)
    #plt.ylim(10**(-16), 10**(-0))
    plt.legend()
    #plt.gca().invert_yaxis()
    plt.show()
    
    
    
def pt_sigeff(taggers):
    plt.figure(figsize=[16,12])
    nbins = 25
    for t in taggers:
        some_tagger = t

    h_sig_total = root.TH1D (f"sig_total",f"sigtotal",nbins,0,3000)
#    MASSBINS = np.linspace(200, 3000, (300 - 40) // 5 + 1, endpoint=True)

    fh(h_sig_total,taggers[some_tagger].signal["fjet_pt"],taggers[some_tagger].signal["EventInfo_mcEventWeight"]*taggers[some_tagger].signal["xsec_weight"])
    a_sig = h2a(h_sig_total)
#    plt.semilogy(np.linspace(0, 3000, 100), a_sig, label="all signal")
    
    for t in taggers:
        h_sig = root.TH1D (f"sig_{taggers[t].name}",f"sig_{taggers[t].name}",nbins,0,3000)
        fh(h_sig,taggers[t].signal_tagged["fjet_pt"],taggers[t].signal_tagged["EventInfo_mcEventWeight"]*taggers[t].signal_tagged["xsec_weight"])
        sig = h2a(h_sig)
        plt.plot(np.linspace(0, 3000, nbins),sig/a_sig, 
                     label="tagged signal {}".format(taggers[t].name))

    

    plt.ylabel("Reweighted event counts")
    plt.xlabel("Jet pT $[GeV]$")
    #plt.xlim(5,500)
    #plt.ylim(10**(-16), 10**(-0))
    plt.legend()
    #plt.gca().invert_yaxis()
    plt.show()

def weights(tagger):
    plt.figure(figsize=[16,8])
    kwargs = dict(alpha = 0.75, bins = 50, density = True, stacked = True,range=(0,1))

    plt.hist(tagger.signal["fjet_weight_pt_dR"],  ** kwargs, label="Signal")
    plt.hist(tagger.bg["fjet_weight_pt_dR"], ** kwargs, label="Background")
    plt.legend(fontsize=30)
    plt.show()

