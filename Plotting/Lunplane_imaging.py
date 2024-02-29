import ROOT

# load a ROOT file
#file = ROOT.TFile.Open("/eos/user/r/rvinasco/JetTagging/LundNet_data/full_splitings/5_Signal_and_BKG/user.rvinasco.30879688._000001.tree.root_train.root")


#file = ROOT.TFile.Open("/eos/user/t/tmlinare/Lund_tagger/ljptagger/Models/lunplane_plots/backgroundtrain.root")
#file = ROOT.TFile.Open("/eos/user/t/tmlinare/Lund_tagger/ljptagger/Models/lunplane_plots/backgroundtest.root")
#file = ROOT.TFile.Open("/eos/user/t/tmlinare/Lund_tagger/ljptagger/Models/lunplane_plots/signaltrain.root")
#file = ROOT.TFile.Open("/eos/user/t/tmlinare/Lund_tagger/ljptagger/Models/lunplane_plots/signaltest.root")
file = ROOT.TFile.Open("/eos/user/t/tmlinare/Lund_tagger/ljptagger/Models/Trainingfile/user.teris.10.1.tree.root_train.root")



# load a tree from the ROOT file
tree = file.Get("FlatSubstructureJetTree")
#tree = file.Get("FlatSubstructureJetTree")

# plot the variable "UFO_jetLundz" versus "UFO_jetLundDeltaR" (from the tree) and save the plot in a file
c1 = ROOT.TCanvas()
ROOT.gStyle.SetOptStat(0)
#drawing ln(1/z) graph
tree.Draw("log(1/UFO_jetLundz[0]):log(1/UFO_jetLundDeltaR[0])>>hlp(40,0,5,40,0.5,5)","","colz")

#tree.Draw("log(UFO_jetLundKt[0]):log(0.4/UFO_jetLundDeltaR[0])>>hlp(40,0,5,40,0,5)","","colz")
hlp = ROOT.gDirectory.Get("hlp")
hlp.SetTitle("")
hlp.GetXaxis().SetTitle("ln(R/deltaR)")
hlp.GetYaxis().SetTitle("ln(1/z)")
#c1.title("QCD Lundplane")
#c1.legend()

#c1.SaveAs("/eos/user/t/tmlinare/Lund_tagger/ljptagger/Models/lundplane_imaging_script/Lund_plot_background_train_z.png")
#c1.SaveAs("/eos/user/t/tmlinare/Lund_tagger/ljptagger/Models/lundplane_imaging_script/Lund_plot_background_test_z.png")
#c1.SaveAs("/eos/user/t/tmlinare/Lund_tagger/ljptagger/Models/lundplane_imaging_script/Lund_plot_signal_train_z.png")
#c1.SaveAs("/eos/user/t/tmlinare/Lund_tagger/ljptagger/Models/lundplane_imaging_script/Lund_plot_signal_test_z.png")
c1.SaveAs("/eos/user/t/tmlinare/Lund_tagger/ljptagger/Models/lundplane_imaging_script/Lund_plot_mixed_train_z_nolog.png")


c1 = ROOT.TCanvas()
ROOT.gStyle.SetOptStat(0)
tree.Draw("log(UFO_jetLundKt[0]):log(1/UFO_jetLundDeltaR[0])>>hlp(40,0,5,40,0,5)","","colz")
hlp = ROOT.gDirectory.Get("hlp")
hlp.SetTitle("")
hlp.GetXaxis().SetTitle("ln(R/deltaR)")
hlp.GetYaxis().SetTitle("ln(kt)")
#c1.title("QCD Lundplane")
#c1.legend()
#c1.SaveAs("/eos/user/t/tmlinare/Lund_tagger/ljptagger/Models/lundplane_imaging_script/Lund_plot_background_train_kt.png")
#c1.SaveAs("/eos/user/t/tmlinare/Lund_tagger/ljptagger/Models/lundplane_imaging_script/Lund_plot_background_test_kt.png")
#c1.SaveAs("/eos/user/t/tmlinare/Lund_tagger/ljptagger/Models/lundplane_imaging_script/Lund_plot_signal_train_kt.png")
#c1.SaveAs("/eos/user/t/tmlinare/Lund_tagger/ljptagger/Models/lundplane_imaging_script/Lund_plot_signal_test_kt.png")
c1.SaveAs("/eos/user/t/tmlinare/Lund_tagger/ljptagger/Models/lundplane_imaging_script/Lund_plot_mixed_train_kt_nolog.png")