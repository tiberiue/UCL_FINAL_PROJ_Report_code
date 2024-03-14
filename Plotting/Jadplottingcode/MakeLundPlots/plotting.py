import ROOT
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, mode, iqr
import math
import argparse
from rootplotting import ap
from rootplotting.tools import *
from root_numpy import fill_hist

def DrawLundPlaneRatio(hist1, hist2, labelx='', labely = ''):

    c = ap.canvas(batch=True, size=(600,600))
    c.pads()[0]._bare().SetRightMargin(0.2)
    # c.pads()[0]._bare().SetLogz()

    hist1.GetZaxis().SetTitle("Number of emissions")
    # hist1.GetZaxis().SetRangeUser(0, 2)

    hist1.Scale(1./hist1.Integral())
    hist2.Scale(1./hist2.Integral())
    ratio = hist1.Clone()

    ratio.Divide(hist2)
    ratio.GetZaxis().SetRangeUser(0.7, 1.3)

    c.hist2d(ratio, option='AXIS')
    c.hist2d(ratio,         option='COLZ')
    c.hist2d(ratio, option='AXIS')
    c.ylim(-1, 10)
    c.xlabel(labelx)
    c.ylabel(labely)


    c.text(["#sqrt{s} = 13 TeV, #it{W} tagging",
             "#scale[0.85]{anti-k_{t} R=1.0 UFO Soft-Drop CS+SK jets}",
              "#scale[0.85]{QCD jets}"
             ], qualifier='Simulation Preliminary')
    return c

def DrawLundPlane(hist, labelx='', labely = '', WorWCD=''):

    # ml response vs input variables raw
    c = ap.canvas(batch=True, size=(600,600))
    c.pads()[0]._bare().SetRightMargin(0.2)
    c.pads()[0]._bare().SetLogz()

    # xaxis = np.linspace(-1,  3, 100 + 1, endpoint=True)
    # yaxis = np.linspace(0, 10,  100 + 1, endpoint=True)

    # h1_backdrop = ROOT.TH2F('', "", 1, np.array([xaxis[0], xaxis[-1]]), 1, np.array([yaxis[0], 0.75* yaxis[-1] ]))
    # h1          = ROOT.TH2F('', '', len(xaxis) - 1, xaxis, len(yaxis) - 1, yaxis)

    # mesh = np.vstack((energy, resp)).T
    # fill_hist(h1, mesh)

    # h1.GetZaxis().SetRangeUser(1, 1e3)
    # h1.GetZaxis().SetRangeUser(1, 1e3)
    # hist.GetZaxis().SetTitle("d^{2}N_{emissions}/(dln(k_{T})dln(1/#Delta R))")
    hist.GetZaxis().SetTitle("Number of emissions")
    # hist.GetZaxis().SetRangeUser(1, 1e4)

    c.hist2d(hist, option='AXIS')
    c.hist2d(hist,         option='COLZ')
    c.hist2d(hist, option='AXIS')
    c.ylim(-1, 10)
    c.xlabel(labelx)
    c.ylabel(labely)
    if WorWCD=="Z":
        c.text(["#sqrt{s} = 13 TeV, #it{top} tagging",
                     "#scale[0.85]{anti-k_{t} R=1.0 UFO Soft-Drop CS+SK jets}",
                     "#scale[0.85]{top jets}"
                ], qualifier='Simulation Preliminary')
    if WorWCD=="QCD":
        c.text(["#sqrt{s} = 13 TeV, #it{top} tagging",
                     "#scale[0.85]{anti-k_{t} R=1.0 UFO Soft-Drop CS+SK jets}",
                     "#scale[0.85]{QCD jets}"
                ], qualifier='Simulation Preliminary')
    return c


def main():
    ROOT.gStyle.SetPalette(ROOT.kBird)

    filetop = ROOT.TFile.Open("lundplanewprime.root")
    ZLundplane   = filetop.Get("hLund_kt_dR")
    ZLundplane_lowpT   = filetop.Get("hLund_kt_dR_pt_200_500")
    ZLundplane_highpT   = filetop.Get("hLund_kt_dR_pt_1000_3000")
    #
    fileQCD = ROOT.TFile.Open("lundplaneqcd.root")
    QCDLundplane   = fileQCD.Get("hLund_kt_dR")
    QCDLundplane_lowpT   = fileQCD.Get("hLund_kt_dR_pt_200_500")
    QCDLundplane_highpT   = fileQCD.Get("hLund_kt_dR_pt_1000_3000")


    c = DrawLundPlane(ZLundplane, labelx='ln(1/#Delta R)', labely = 'ln(k_{T})', WorWCD='Z')
    c.save("./Plots/WLundplane.png")
    c.save("./Plots/WLundplane.pdf")
    c.save("./Plots/WLundplane.eps")
    #
    c = DrawLundPlane(QCDLundplane, labelx='ln(1/#Delta R)', labely = 'ln(k_{T})', WorWCD='QCD')
    c.save("./Plots/QCDLundplane.png")
    c.save("./Plots/QCDLundplane.pdf")
    c.save("./Plots/QCDLundplane.eps")

    c = DrawLundPlane(ZLundplane_lowpT, labelx='ln(1/#Delta R)', labely = 'ln(k_{T})', WorWCD='Z')
    c.save("./Plots/WLundplane_lowpT.png")
    c.save("./Plots/WLundplane_lowpT.pdf")
    c.save("./Plots/WLundplane_lowpT.eps")

    c = DrawLundPlane(QCDLundplane_lowpT, labelx='ln(1/#Delta R)', labely = 'ln(k_{T})', WorWCD='QCD')
    c.save("./Plots/QCDLundplane_lowpT.png")
    c.save("./Plots/QCDLundplane_lowpT.pdf")
    c.save("./Plots/QCDLundplane_lowpT.eps")

    c = DrawLundPlane(ZLundplane_highpT, labelx='ln(1/#Delta R)', labely = 'ln(k_{T})', WorWCD='Z')
    c.save("./Plots/WLundplane_highpT.png")
    c.save("./Plots/WLundplane_highpT.pdf")
    c.save("./Plots/WLundplane_highpT.eps")

    c = DrawLundPlane(QCDLundplane_highpT, labelx='ln(1/#Delta R)', labely = 'ln(k_{T})', WorWCD='QCD')
    c.save("./Plots/QCDLundplane_highpT.png")
    c.save("./Plots/QCDLundplane_highpT.pdf")
    c.save("./Plots/QCDLundplane_highpT.eps")

    # Pythia
    # filePythia = ROOT.TFile.Open("lundplane_Pythia.root")
    # QCDLundplanePythia = filePythia.Get("hLund_kt_dR_bkg")
    # c = DrawLundPlane(QCDLundplanePythia, labelx='ln(1/#Delta R)', labely = 'ln(k_{T})', WorWCD='QCD')
    # c.save("./QCDLundplanePythia.png")
    # # SherpaLund
    # fileLund = ROOT.TFile.Open("lundplane_SherpaLund.root")
    # QCDLundplaneSherpaLund = fileLund.Get("hLund_kt_dR_bkg")
    # c = DrawLundPlane(QCDLundplaneSherpaLund, labelx='ln(1/#Delta R)', labely = 'ln(k_{T})', WorWCD='QCD')
    # c.save("./QCDLundplaneSherpaLund.png")
    # # SherpaAHADIC
    # fileAHADIC = ROOT.TFile.Open("lundplane_SherpaAHADIC.root")
    # QCDLundplaneSherpaAHADIC = fileAHADIC.Get("hLund_kt_dR_bkg")
    # c = DrawLundPlane(QCDLundplaneSherpaAHADIC, labelx='ln(1/#Delta R)', labely = 'ln(k_{T})', WorWCD='QCD')
    # c.save("./QCDLundplaneSherpaAHADIC.png")
    # # HerwigAngular
    # fileAngular = ROOT.TFile.Open("lundplane_HerwigAngular.root")
    # QCDLundplaneHerwigAngular = fileAngular.Get("hLund_kt_dR_bkg")
    # c = DrawLundPlane(QCDLundplaneHerwigAngular, labelx='ln(1/#Delta R)', labely = 'ln(k_{T})', WorWCD='QCD')
    # c.save("./QCDLundplaneHerwigAngular.png")
    # # HerwigDipole
    # fileDipole = ROOT.TFile.Open("lundplane_HerwigDipole.root")
    # QCDLundplaneHerwigDipole = fileDipole.Get("hLund_kt_dR_bkg")
    # c = DrawLundPlane(QCDLundplaneHerwigDipole, labelx='ln(1/#Delta R)', labely = 'ln(k_{T})', WorWCD='QCD')
    # c.save("./QCDLundplaneHerwigDipole.png")

    #
    # c = DrawLundPlaneRatio(QCDLundplaneHerwigAngular, QCDLundplaneHerwigDipole, labelx='ln(1/#Delta R)', labely = 'ln(k_{T})')
    # c.save("./QCDLundplane_AngularDipole.png")
    # c = DrawLundPlaneRatio(QCDLundplaneSherpaLund, QCDLundplaneSherpaAHADIC, labelx='ln(1/#Delta R)', labely = 'ln(k_{T})')
    # c.save("./QCDLundplane_AHADICLund.png")
    # c = DrawLundPlaneRatio(QCDLundplaneSherpaLund, QCDLundplanePythia , labelx='ln(1/#Delta R)', labely = 'ln(k_{T})')
    # c.save("./QCDLundplane_PythiaLund.png")



# Main function call.
if __name__ == '__main__':
    main()
    pass
