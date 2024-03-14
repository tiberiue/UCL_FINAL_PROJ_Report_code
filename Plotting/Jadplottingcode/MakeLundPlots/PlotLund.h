//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Fri Feb 23 09:36:37 2024 by ROOT version 6.30/02
// from TTree FlatSubstructureJetTree/FlatSubstructureJetTree
// found on file: /data/jmsardain/LJPTagger/FullSplittings/raw_data/user.rvinasco..364710.Flattener_UFO_Soft_drop_Full_splittingsLJP_tree.root/user.rvinasco.30617199._000001.tree.root
//////////////////////////////////////////////////////////

#ifndef PlotLund_h
#define PlotLund_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>

// Header file for the classes stored in the TTree if any.
#include "vector"
#include "vector"
using namespace std;
class PlotLund {
public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain
   Int_t           fCurrent; //!current Tree number in a TChain

// Fixed size dimensions of array or collections stored in the TTree if any.

   // Declaration of leaf types
   Int_t           DSID;
   Float_t         mcWeight;
   Float_t         UFOSD_jetPt;
   Float_t         UFOSD_jetEta;
   Float_t         UFOSD_jetPhi;
   Float_t         UFOSD_jetM;
   Float_t         UFO_jetPt;
   Float_t         UFO_jetEta;
   Float_t         UFO_jetPhi;
   Float_t         UFO_jetM;
   vector<float>   *UFO_jetLundKt;
   vector<float>   *UFO_jetLundz;
   vector<float>   *UFO_jetLundDeltaR;
   vector<int>     *UFO_edge1;
   vector<int>     *UFO_edge2;
   Bool_t          Akt10UFOJet_dRMatched_truthJet;
   Float_t         Akt10UFOJet_dRmatched_particle_flavor;
   Float_t         Akt10UFOJet_dRmatched_particle_dR;
   Float_t         Akt10UFOJet_dRmatched_particle_dR_top_W_matched;
   Float_t         Akt10UFOJet_ungroomedParent_GhostBHadronsFinalCount;
   Float_t         Akt10UFOJet_ungroomed_truthJet_Split23;
   Float_t         Akt10UFOJet_ungroomed_truthJet_pt;
   Float_t         Akt10UFOJet_ungroomed_truthJet_m;
   Int_t           Akt10UFOJet_jetNumConst;
   Int_t           Akt10UFOJet_GhostBHadronsFinalCount;
   Float_t         Akt10UFOJet_Split12;
   Float_t         Akt10UFOJet_ungroomed_truthJet_Split12;
   Float_t         UFO_jetTaggerResultMassLow50;
   Float_t         UFO_jetTaggerResultMassHigh50;
   Float_t         UFO_jetTagResultD250;
   Float_t         UFO_jetTagResultNtrk50;
   Float_t         UFO_jetTaggerResultMassLow80;
   Float_t         UFO_jetTaggerResultMassHigh80;
   Float_t         UFO_jetTagResultD280;
   Float_t         UFO_jetTagResultNtrk80;
   Float_t         UFO_jetTagResultScoreInclusive50;
   Float_t         UFO_jetTagResultScoreInclusive80;
   Float_t         UFO_jetTagResultScoreContained50;
   Float_t         UFO_jetTagResultScoreContained80;
   Float_t         UFO_DNNScoreInclusive;
   Float_t         UFO_DNNScoreContained;
   Float_t         UFO_D2;
   Int_t           UFO_Ntrk;
   Float_t         UFO_Dip12;
   Float_t         UFO_KtDR;
   Float_t         UFO_PlanarFlow;
   Float_t         UFO_C2;
   Float_t         UFO_Tau12_wta;
   Float_t         UFO_FoxWolfram20;
   Float_t         UFO_Angularity;
   Float_t         UFO_Aplanarity;
   Float_t         UFO_ZCut12;
   Float_t         fjet_testing_weight_pt;
   Float_t         fjet_testing_weight_pt_dR;
   Float_t         fjet_testing_weight_pt_m140;

   // List of branches
   TBranch        *b_DSID;   //!
   TBranch        *b_mcWeight;   //!
   TBranch        *b_UFOSD_jetPt;   //!
   TBranch        *b_UFOSD_jetEta;   //!
   TBranch        *b_UFOSD_jetPhi;   //!
   TBranch        *b_UFOSD_jetM;   //!
   TBranch        *b_UFO_jetPt;   //!
   TBranch        *b_UFO_jetEta;   //!
   TBranch        *b_UFO_jetPhi;   //!
   TBranch        *b_UFO_jetM;   //!
   TBranch        *b_UFO_jetLundKt;   //!
   TBranch        *b_UFO_jetLundz;   //!
   TBranch        *b_UFO_jetLundDeltaR;   //!
   TBranch        *b_UFO_edge1;   //!
   TBranch        *b_UFO_edge2;   //!
   TBranch        *b_Akt10UFOJet_dRMatched_truthJet;   //!
   TBranch        *b_Akt10UFOJet_dRmatched_particle_flavor;   //!
   TBranch        *b_Akt10UFOJet_dRmatched_particle_dR;   //!
   TBranch        *b_Akt10UFOJet_dRmatched_particle_dR_top_W_matched;   //!
   TBranch        *b_Akt10UFOJet_ungroomedParent_GhostBHadronsFinalCount;   //!
   TBranch        *b_Akt10UFOJet_ungroomed_truthJet_Split23;   //!
   TBranch        *b_Akt10UFOJet_ungroomed_truthJet_pt;   //!
   TBranch        *b_Akt10UFOJet_ungroomed_truthJet_m;   //!
   TBranch        *b_Akt10UFOJet_jetNumConst;   //!
   TBranch        *b_Akt10UFOJet_GhostBHadronsFinalCount;   //!
   TBranch        *b_Akt10UFOJet_Split12;   //!
   TBranch        *b_Akt10UFOJet_ungroomed_truthJet_Split12;   //!
   TBranch        *b_UFO_jetTaggerResultMassLow50;   //!
   TBranch        *b_UFO_jetTaggerResultMassHigh50;   //!
   TBranch        *b_UFO_jetTagResultD250;   //!
   TBranch        *b_UFO_jetTagResultNtrk50;   //!
   TBranch        *b_UFO_jetTaggerResultMassLow80;   //!
   TBranch        *b_UFO_jetTaggerResultMassHigh80;   //!
   TBranch        *b_UFO_jetTagResultD280;   //!
   TBranch        *b_UFO_jetTagResultNtrk80;   //!
   TBranch        *b_UFO_jetTagResultScoreInclusive50;   //!
   TBranch        *b_UFO_jetTagResultScoreInclusive80;   //!
   TBranch        *b_UFO_jetTagResultScoreContained50;   //!
   TBranch        *b_UFO_jetTagResultScoreContained80;   //!
   TBranch        *b_UFO_DNNScoreInclusive;   //!
   TBranch        *b_UFO_DNNScoreContained;   //!
   TBranch        *b_UFO_D2;   //!
   TBranch        *b_UFO_Ntrk;   //!
   TBranch        *b_UFO_Dip12;   //!
   TBranch        *b_UFO_KtDR;   //!
   TBranch        *b_UFO_PlanarFlow;   //!
   TBranch        *b_UFO_C2;   //!
   TBranch        *b_UFO_Tau12_wta;   //!
   TBranch        *b_UFO_FoxWolfram20;   //!
   TBranch        *b_UFO_Angularity;   //!
   TBranch        *b_UFO_Aplanarity;   //!
   TBranch        *b_UFO_ZCut12;   //!
   TBranch        *b_fjet_testing_weight_pt;   //!
   TBranch        *b_fjet_testing_weight_pt_dR;   //!
   TBranch        *b_fjet_testing_weight_pt_m140;   //!

   PlotLund(TTree *tree=0);
   virtual ~PlotLund();
   virtual Int_t    Cut(Long64_t entry);
   virtual Int_t    GetEntry(Long64_t entry);
   virtual Long64_t LoadTree(Long64_t entry);
   virtual void     Init(TTree *tree);
   virtual void     Loop(TString SigOrBkg);
   virtual Bool_t   Notify();
   virtual void     Show(Long64_t entry = -1);
};

#endif

#ifdef PlotLund_cxx
PlotLund::PlotLund(TTree *tree) : fChain(0)
{
// if parameter tree is not specified (or zero), connect the file
// used to generate this class and read the Tree.
   if (tree == 0) {
      TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject("/data/jmsardain/LJPTagger/FullSplittings/raw_data/user.rvinasco..364710.Flattener_UFO_Soft_drop_Full_splittingsLJP_tree.root/user.rvinasco.30617199._000001.tree.root");
      if (!f || !f->IsOpen()) {
         f = new TFile("/data/jmsardain/LJPTagger/FullSplittings/raw_data/user.rvinasco..364710.Flattener_UFO_Soft_drop_Full_splittingsLJP_tree.root/user.rvinasco.30617199._000001.tree.root");
      }
      f->GetObject("FlatSubstructureJetTree",tree);

   }
   Init(tree);
}

PlotLund::~PlotLund()
{
   if (!fChain) return;
   delete fChain->GetCurrentFile();
}

Int_t PlotLund::GetEntry(Long64_t entry)
{
// Read contents of entry.
   if (!fChain) return 0;
   return fChain->GetEntry(entry);
}
Long64_t PlotLund::LoadTree(Long64_t entry)
{
// Set the environment to read one entry
   if (!fChain) return -5;
   Long64_t centry = fChain->LoadTree(entry);
   if (centry < 0) return centry;
   if (fChain->GetTreeNumber() != fCurrent) {
      fCurrent = fChain->GetTreeNumber();
      Notify();
   }
   return centry;
}

void PlotLund::Init(TTree *tree)
{
   // The Init() function is called when the selector needs to initialize
   // a new tree or chain. Typically here the branch addresses and branch
   // pointers of the tree will be set.
   // It is normally not necessary to make changes to the generated
   // code, but the routine can be extended by the user if needed.
   // Init() will be called many times when running on PROOF
   // (once per file to be processed).

   // Set object pointer
   UFO_jetLundKt = 0;
   UFO_jetLundz = 0;
   UFO_jetLundDeltaR = 0;
   UFO_edge1 = 0;
   UFO_edge2 = 0;
   // Set branch addresses and branch pointers
   if (!tree) return;
   fChain = tree;
   fCurrent = -1;
   fChain->SetMakeClass(1);

   fChain->SetBranchAddress("DSID", &DSID, &b_DSID);
   fChain->SetBranchAddress("mcWeight", &mcWeight, &b_mcWeight);
   fChain->SetBranchAddress("UFOSD_jetPt", &UFOSD_jetPt, &b_UFOSD_jetPt);
   fChain->SetBranchAddress("UFOSD_jetEta", &UFOSD_jetEta, &b_UFOSD_jetEta);
   fChain->SetBranchAddress("UFOSD_jetPhi", &UFOSD_jetPhi, &b_UFOSD_jetPhi);
   fChain->SetBranchAddress("UFOSD_jetM", &UFOSD_jetM, &b_UFOSD_jetM);
   fChain->SetBranchAddress("UFO_jetPt", &UFO_jetPt, &b_UFO_jetPt);
   fChain->SetBranchAddress("UFO_jetEta", &UFO_jetEta, &b_UFO_jetEta);
   fChain->SetBranchAddress("UFO_jetPhi", &UFO_jetPhi, &b_UFO_jetPhi);
   fChain->SetBranchAddress("UFO_jetM", &UFO_jetM, &b_UFO_jetM);
   fChain->SetBranchAddress("UFO_jetLundKt", &UFO_jetLundKt, &b_UFO_jetLundKt);
   fChain->SetBranchAddress("UFO_jetLundz", &UFO_jetLundz, &b_UFO_jetLundz);
   fChain->SetBranchAddress("UFO_jetLundDeltaR", &UFO_jetLundDeltaR, &b_UFO_jetLundDeltaR);
   fChain->SetBranchAddress("UFO_edge1", &UFO_edge1, &b_UFO_edge1);
   fChain->SetBranchAddress("UFO_edge2", &UFO_edge2, &b_UFO_edge2);
   fChain->SetBranchAddress("Akt10UFOJet_dRMatched_truthJet", &Akt10UFOJet_dRMatched_truthJet, &b_Akt10UFOJet_dRMatched_truthJet);
   fChain->SetBranchAddress("Akt10UFOJet_dRmatched_particle_flavor", &Akt10UFOJet_dRmatched_particle_flavor, &b_Akt10UFOJet_dRmatched_particle_flavor);
   fChain->SetBranchAddress("Akt10UFOJet_dRmatched_particle_dR", &Akt10UFOJet_dRmatched_particle_dR, &b_Akt10UFOJet_dRmatched_particle_dR);
   fChain->SetBranchAddress("Akt10UFOJet_dRmatched_particle_dR_top_W_matched", &Akt10UFOJet_dRmatched_particle_dR_top_W_matched, &b_Akt10UFOJet_dRmatched_particle_dR_top_W_matched);
   fChain->SetBranchAddress("Akt10UFOJet_ungroomedParent_GhostBHadronsFinalCount", &Akt10UFOJet_ungroomedParent_GhostBHadronsFinalCount, &b_Akt10UFOJet_ungroomedParent_GhostBHadronsFinalCount);
   fChain->SetBranchAddress("Akt10UFOJet_ungroomed_truthJet_Split23", &Akt10UFOJet_ungroomed_truthJet_Split23, &b_Akt10UFOJet_ungroomed_truthJet_Split23);
   fChain->SetBranchAddress("Akt10UFOJet_ungroomed_truthJet_pt", &Akt10UFOJet_ungroomed_truthJet_pt, &b_Akt10UFOJet_ungroomed_truthJet_pt);
   fChain->SetBranchAddress("Akt10UFOJet_ungroomed_truthJet_m", &Akt10UFOJet_ungroomed_truthJet_m, &b_Akt10UFOJet_ungroomed_truthJet_m);
   fChain->SetBranchAddress("Akt10UFOJet_jetNumConst", &Akt10UFOJet_jetNumConst, &b_Akt10UFOJet_jetNumConst);
   fChain->SetBranchAddress("Akt10UFOJet_GhostBHadronsFinalCount", &Akt10UFOJet_GhostBHadronsFinalCount, &b_Akt10UFOJet_GhostBHadronsFinalCount);
   fChain->SetBranchAddress("Akt10UFOJet_Split12", &Akt10UFOJet_Split12, &b_Akt10UFOJet_Split12);
   fChain->SetBranchAddress("Akt10UFOJet_ungroomed_truthJet_Split12", &Akt10UFOJet_ungroomed_truthJet_Split12, &b_Akt10UFOJet_ungroomed_truthJet_Split12);
   fChain->SetBranchAddress("UFO_jetTaggerResultMassLow50", &UFO_jetTaggerResultMassLow50, &b_UFO_jetTaggerResultMassLow50);
   fChain->SetBranchAddress("UFO_jetTaggerResultMassHigh50", &UFO_jetTaggerResultMassHigh50, &b_UFO_jetTaggerResultMassHigh50);
   fChain->SetBranchAddress("UFO_jetTagResultD250", &UFO_jetTagResultD250, &b_UFO_jetTagResultD250);
   fChain->SetBranchAddress("UFO_jetTagResultNtrk50", &UFO_jetTagResultNtrk50, &b_UFO_jetTagResultNtrk50);
   fChain->SetBranchAddress("UFO_jetTaggerResultMassLow80", &UFO_jetTaggerResultMassLow80, &b_UFO_jetTaggerResultMassLow80);
   fChain->SetBranchAddress("UFO_jetTaggerResultMassHigh80", &UFO_jetTaggerResultMassHigh80, &b_UFO_jetTaggerResultMassHigh80);
   fChain->SetBranchAddress("UFO_jetTagResultD280", &UFO_jetTagResultD280, &b_UFO_jetTagResultD280);
   fChain->SetBranchAddress("UFO_jetTagResultNtrk80", &UFO_jetTagResultNtrk80, &b_UFO_jetTagResultNtrk80);
   fChain->SetBranchAddress("UFO_jetTagResultScoreInclusive50", &UFO_jetTagResultScoreInclusive50, &b_UFO_jetTagResultScoreInclusive50);
   fChain->SetBranchAddress("UFO_jetTagResultScoreInclusive80", &UFO_jetTagResultScoreInclusive80, &b_UFO_jetTagResultScoreInclusive80);
   fChain->SetBranchAddress("UFO_jetTagResultScoreContained50", &UFO_jetTagResultScoreContained50, &b_UFO_jetTagResultScoreContained50);
   fChain->SetBranchAddress("UFO_jetTagResultScoreContained80", &UFO_jetTagResultScoreContained80, &b_UFO_jetTagResultScoreContained80);
   fChain->SetBranchAddress("UFO_DNNScoreInclusive", &UFO_DNNScoreInclusive, &b_UFO_DNNScoreInclusive);
   fChain->SetBranchAddress("UFO_DNNScoreContained", &UFO_DNNScoreContained, &b_UFO_DNNScoreContained);
   fChain->SetBranchAddress("UFO_D2", &UFO_D2, &b_UFO_D2);
   fChain->SetBranchAddress("UFO_Ntrk", &UFO_Ntrk, &b_UFO_Ntrk);
   fChain->SetBranchAddress("UFO_Dip12", &UFO_Dip12, &b_UFO_Dip12);
   fChain->SetBranchAddress("UFO_KtDR", &UFO_KtDR, &b_UFO_KtDR);
   fChain->SetBranchAddress("UFO_PlanarFlow", &UFO_PlanarFlow, &b_UFO_PlanarFlow);
   fChain->SetBranchAddress("UFO_C2", &UFO_C2, &b_UFO_C2);
   fChain->SetBranchAddress("UFO_Tau12_wta", &UFO_Tau12_wta, &b_UFO_Tau12_wta);
   fChain->SetBranchAddress("UFO_FoxWolfram20", &UFO_FoxWolfram20, &b_UFO_FoxWolfram20);
   fChain->SetBranchAddress("UFO_Angularity", &UFO_Angularity, &b_UFO_Angularity);
   fChain->SetBranchAddress("UFO_Aplanarity", &UFO_Aplanarity, &b_UFO_Aplanarity);
   fChain->SetBranchAddress("UFO_ZCut12", &UFO_ZCut12, &b_UFO_ZCut12);
   fChain->SetBranchAddress("fjet_testing_weight_pt", &fjet_testing_weight_pt, &b_fjet_testing_weight_pt);
   fChain->SetBranchAddress("fjet_testing_weight_pt_dR", &fjet_testing_weight_pt_dR, &b_fjet_testing_weight_pt_dR);
   fChain->SetBranchAddress("fjet_testing_weight_pt_m140", &fjet_testing_weight_pt_m140, &b_fjet_testing_weight_pt_m140);
   Notify();
}

Bool_t PlotLund::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normally not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   return kTRUE;
}

void PlotLund::Show(Long64_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}
Int_t PlotLund::Cut(Long64_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return 1;
}
#endif // #ifdef PlotLund_cxx
