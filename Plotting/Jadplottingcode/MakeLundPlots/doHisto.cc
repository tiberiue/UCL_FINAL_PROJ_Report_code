#include <iostream>
#include "TString.h"
#include "PlotLund.h"
#include "TChain.h"
#include "TFile.h"
#include "TTree.h"
#include "TROOT.h"
#include <string>
#include "AtlasStyle.h"
#include "AtlasLabels.h"
#include "AtlasUtils.h"

using namespace std ;

int main(int argc, char* argv[]){
// int main(){
	SetAtlasStyle();
	std::string name(argv[1]);
	// data
	// TString theLink = "data/user.zhcui.34376381._000001.ANALYSIS.root";
	TString SigOrBkg = TString(name.c_str());
	TString theLink;
	if (SigOrBkg == "qcd"){
		theLink = "/eos/user/t/tmlinare/Lund_tagger/ljptagger/Models/OtherMC/Herwigdipole/user.teris.Herwigdipole.root";
	}
	if (SigOrBkg == "wprime"){
		theLink = "/eos/user/r/rvinasco/JetTagging/Other_MC_Samples/splitted_SherpaSignal_files/user.rvinasco.33272437._000002.tree.rootpart_1.root";
	}
	if (SigOrBkg == "zprime"){
		theLink = "/eos/user/t/tmlinare/Lund_tagger/ljptagger/Models/Trainingfile/user.teris.zprime.tree.root_train.root";
	}


	TChain * myChain = new TChain( "FlatSubstructureJetTree" ) ;

	cout << theLink << endl ;

	myChain->Add( theLink );
	cout << "my chain = " << myChain->GetEntries() << endl ;

	// gROOT->LoadMacro("/afs/cern.ch/work/j/jzahredd/EarlyRun3/HistoMaker.C+");

	PlotLund * myAnalysis ;
	myAnalysis =  new PlotLund( myChain ) ;
	myAnalysis->Loop(SigOrBkg);
	return 0;

}
