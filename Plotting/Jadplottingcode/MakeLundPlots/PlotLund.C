#define PlotLund_cxx
#include "PlotLund.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TMath.h>
#include <vector>
#include <iostream>

void PlotLund::Loop(TString SigOrBkg){

  if (fChain == 0) return;

  TFile*f = new TFile("lundplane"+SigOrBkg+".root", "recreate");
  // inclusive
  TH2D *hLund_z_dR  = new TH2D("", "signal:1/z:1/dr", 200, 0, 5, 200, 0, 5);
  TH2D *hLund_kt_dR  = new TH2D("", "signal:kt:1/dr", 200, 0, 5, 200, -1, 10);
  // pt 200, 500
  TH2D *hLund_z_dR_pt_200_500 = new TH2D("", "signal:1/z:1/dr", 200, 0, 5, 200, 0, 5);
  TH2D *hLund_kt_dR_pt_200_500   = new TH2D("", "signal:kt:1/dr", 200, 0, 5, 200, -1, 8);

  // pt 500, 1000
  TH2D *hLund_z_dR_pt_500_1000  = new TH2D("", "signal:1/z:1/dr", 200, 0, 5, 200, 0, 5);
  TH2D *hLund_kt_dR_pt_500_1000   = new TH2D("", "signal:kt:1/dr",  200, 0, 5, 200, -1, 8);

  // pt 1000, 3000
  TH2D *hLund_z_dR_pt_1000_3000  = new TH2D("", "signal:1/z:1/dr", 200, 0, 5, 200, 0, 5);
  TH2D *hLund_kt_dR_pt_1000_3000   = new TH2D("", "signal:kt:1/dr", 200, 0, 5, 200, -1, 8);



  Long64_t nentries = fChain->GetEntriesFast();
  Long64_t nbytes = 0, nb = 0;
  for (Long64_t jentry=0; jentry<nentries;jentry++) {
    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0) break;
    nb = fChain->GetEntry(jentry);   nbytes += nb;

    if(jentry%100000==0) std::cout << "Event#" << jentry << std::endl;

    for(unsigned int i=0; i<UFO_jetLundKt->size(); i++){

      if(SigOrBkg=="wprime" && Akt10UFOJet_GhostBHadronsFinalCount==0){
        hLund_z_dR->Fill( TMath::Log(1./UFO_jetLundDeltaR->at(i)),  TMath::Log(1/UFO_jetLundz->at(i)));
        hLund_kt_dR->Fill( TMath::Log(1./UFO_jetLundDeltaR->at(i)), TMath::Log(UFO_jetLundKt->at(i)));
        if (UFOSD_jetPt>200 && UFOSD_jetPt<500){
          hLund_z_dR_pt_200_500->Fill( TMath::Log(1./UFO_jetLundDeltaR->at(i)),  TMath::Log(1/UFO_jetLundz->at(i)));
          hLund_kt_dR_pt_200_500->Fill( TMath::Log(1./UFO_jetLundDeltaR->at(i)), TMath::Log(UFO_jetLundKt->at(i)));
        }
        if (UFOSD_jetPt>500 && UFOSD_jetPt<1000){
          hLund_z_dR_pt_500_1000->Fill( TMath::Log(1./UFO_jetLundDeltaR->at(i)),  TMath::Log(1/UFO_jetLundz->at(i)));
          hLund_kt_dR_pt_500_1000->Fill( TMath::Log(1./UFO_jetLundDeltaR->at(i)), TMath::Log(UFO_jetLundKt->at(i)));
        }
        if (UFOSD_jetPt>1000 && UFOSD_jetPt<3000){
          hLund_z_dR_pt_1000_3000->Fill( TMath::Log(1./UFO_jetLundDeltaR->at(i)),  TMath::Log(1/UFO_jetLundz->at(i)));
          hLund_kt_dR_pt_1000_3000->Fill( TMath::Log(1./UFO_jetLundDeltaR->at(i)), TMath::Log(UFO_jetLundKt->at(i)));
        }
      }
      // if(SigOrBkg=="zprime" && Akt10UFOJet_GhostBHadronsFinalCount>=1){
        if(SigOrBkg=="zprime"){
        hLund_z_dR->Fill( TMath::Log(1./UFO_jetLundDeltaR->at(i)),  TMath::Log(1/UFO_jetLundz->at(i)));
        hLund_kt_dR->Fill( TMath::Log(1./UFO_jetLundDeltaR->at(i)), TMath::Log(UFO_jetLundKt->at(i)));
        if (UFOSD_jetPt>200 && UFOSD_jetPt<500){
          hLund_z_dR_pt_200_500->Fill( TMath::Log(1./UFO_jetLundDeltaR->at(i)),  TMath::Log(1/UFO_jetLundz->at(i)));
          hLund_kt_dR_pt_200_500->Fill( TMath::Log(1./UFO_jetLundDeltaR->at(i)), TMath::Log(UFO_jetLundKt->at(i)));
        }
        if (UFOSD_jetPt>500 && UFOSD_jetPt<1000){
          hLund_z_dR_pt_500_1000->Fill( TMath::Log(1./UFO_jetLundDeltaR->at(i)),  TMath::Log(1/UFO_jetLundz->at(i)));
          hLund_kt_dR_pt_500_1000->Fill( TMath::Log(1./UFO_jetLundDeltaR->at(i)), TMath::Log(UFO_jetLundKt->at(i)));
        }
        if (UFOSD_jetPt>1000 && UFOSD_jetPt<3000){
          hLund_z_dR_pt_1000_3000->Fill( TMath::Log(1./UFO_jetLundDeltaR->at(i)),  TMath::Log(1/UFO_jetLundz->at(i)));
          hLund_kt_dR_pt_1000_3000->Fill( TMath::Log(1./UFO_jetLundDeltaR->at(i)), TMath::Log(UFO_jetLundKt->at(i)));
        }
      }
      if(SigOrBkg=="qcd"){
        hLund_z_dR->Fill( TMath::Log(1./UFO_jetLundDeltaR->at(i)),  TMath::Log(1/UFO_jetLundz->at(i)));
        hLund_kt_dR->Fill( TMath::Log(1./UFO_jetLundDeltaR->at(i)), TMath::Log(UFO_jetLundKt->at(i)));
        if (UFOSD_jetPt>200 && UFOSD_jetPt<500){
          hLund_z_dR_pt_200_500->Fill( TMath::Log(1./UFO_jetLundDeltaR->at(i)),  TMath::Log(1/UFO_jetLundz->at(i)));
          hLund_kt_dR_pt_200_500->Fill( TMath::Log(1./UFO_jetLundDeltaR->at(i)), TMath::Log(UFO_jetLundKt->at(i)));
        }
        if (UFOSD_jetPt>500 && UFOSD_jetPt<1000){
          hLund_z_dR_pt_500_1000->Fill( TMath::Log(1./UFO_jetLundDeltaR->at(i)),  TMath::Log(1/UFO_jetLundz->at(i)));
          hLund_kt_dR_pt_500_1000->Fill( TMath::Log(1./UFO_jetLundDeltaR->at(i)), TMath::Log(UFO_jetLundKt->at(i)));
        }
        if (UFOSD_jetPt>1000 && UFOSD_jetPt<3000){
          hLund_z_dR_pt_1000_3000->Fill( TMath::Log(1./UFO_jetLundDeltaR->at(i)),  TMath::Log(1/UFO_jetLundz->at(i)));
          hLund_kt_dR_pt_1000_3000->Fill( TMath::Log(1./UFO_jetLundDeltaR->at(i)), TMath::Log(UFO_jetLundKt->at(i)));
        }
      }

    }
  }


  f->cd();
  hLund_z_dR->Write("hLund_z_dR");
  hLund_kt_dR->Write("hLund_kt_dR");
  // pt 200, 500
  hLund_z_dR_pt_200_500->Write("hLund_z_dR_pt_200_500");
  hLund_kt_dR_pt_200_500->Write("hLund_kt_dR_pt_200_500");

  // pt 500, 1000
  hLund_z_dR_pt_500_1000->Write("hLund_z_dR_pt_500_1000");
  hLund_kt_dR_pt_500_1000->Write("hLund_kt_dR_pt_500_1000");

  // pt 1000, 3000
  hLund_z_dR_pt_1000_3000->Write("hLund_z_dR_pt_1000_3000");
  hLund_kt_dR_pt_1000_3000->Write("hLund_kt_dR_pt_1000_3000");
  f->Close();

}
