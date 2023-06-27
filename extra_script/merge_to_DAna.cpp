//
// Created by Yulei on 2023/6/26.
//

#include "TFile.h"
#include "TH1F.h"
#include "TTree.h"

#include <iostream>

using namespace std;

void merge_to_DAna(const std::string& gnn_tag_filepath, const std::string& gnn_rec_filepath, const std::string& ana_filepath) {

    // input files
//    auto gnn_tag_file = new TFile("/Users/avencast/PycharmProjects/trkgnn/workspace/output/out_roots/out_0.root",
//                                  "READ");
//    auto gnn_rec_file = new TFile("/Users/avencast/PycharmProjects/trkgnn/workspace/output/out_roots/out_0.root",
//                                  "READ");
//    auto ana_file = new TFile("/Users/avencast/CLionProjects/darkshine-simulation/workspace/dp_ana.root", "READ");

    // input files
    auto gnn_tag_file = new TFile(gnn_tag_filepath.c_str(), "READ");
    auto gnn_rec_file = new TFile(gnn_rec_filepath.c_str(), "READ");
    auto ana_file = new TFile(ana_filepath.c_str(), "READ");

    // input trees
    TTree *treeA = (TTree *) ana_file->Get("dp");
    TTree *tree_tag = (TTree *) gnn_tag_file->Get("gnn_tracks");
    TTree *tree_rec = (TTree *) gnn_rec_file->Get("gnn_tracks");

    // output file and cloned tree
    TFile *fileOut = new TFile("merged.root", "RECREATE");
    TTree *treeOut = treeA->CloneTree(0); // clone schema only

    auto register_values = [&treeOut](TString prefix, std::vector<std::vector<Double_t>> &values) {
        treeOut->Branch(prefix + "_pi", &values[0]);
        treeOut->Branch(prefix + "_pf", &values[1]);
        treeOut->Branch(prefix + "_vertex_x", &values[2]);
        treeOut->Branch(prefix + "_vertex_y", &values[3]);
        treeOut->Branch(prefix + "_vertex_z", &values[4]);
        treeOut->Branch(prefix + "_end_x", &values[5]);
        treeOut->Branch(prefix + "_end_y", &values[6]);
        treeOut->Branch(prefix + "_end_z", &values[7]);
        treeOut->Branch(prefix + "_no_hits", &values[8]);
    };

    std::vector<std::vector<Double_t>> gnn_tag(9, std::vector<Double_t>(0));
    register_values("gnn_tag", gnn_tag);
    std::vector<std::vector<Double_t>> gnn_rec(9, std::vector<Double_t>(0));
    register_values("gnn_rec", gnn_rec);
    Int_t gnn_tag_no_tracks = 0, gnn_rec_no_tracks = 0;
    treeOut->Branch("gnn_tag_no_tracks", &gnn_tag_no_tracks);
    treeOut->Branch("gnn_rec_no_tracks", &gnn_rec_no_tracks);


    Int_t cur_evt, cur_run;
    Int_t tag_evt, tag_run;
    Int_t rec_evt, rec_run;
    treeA->SetBranchAddress("EventNumber", &cur_evt);
    treeA->SetBranchAddress("RunNumber", &cur_run);
    tree_tag->SetBranchAddress("evt_num", &tag_evt);
    tree_tag->SetBranchAddress("run_num", &tag_run);
    tree_rec->SetBranchAddress("evt_num", &rec_evt);
    tree_rec->SetBranchAddress("run_num", &rec_run);

    auto set_branch_address = [](TTree *t, Int_t &nhits, std::vector<Float_t> &values) {
        t->SetBranchAddress("no_hits", &nhits);
        t->SetBranchAddress("p_i", &values[0]);
        t->SetBranchAddress("p_f", &values[1]);
        t->SetBranchAddress("vertex_x", &values[2]);
        t->SetBranchAddress("vertex_y", &values[3]);
        t->SetBranchAddress("vertex_z", &values[4]);
        t->SetBranchAddress("end_x", &values[5]);
        t->SetBranchAddress("end_y", &values[6]);
        t->SetBranchAddress("end_z", &values[7]);
    };

    Int_t tag_nhits;
    std::vector<Float_t> tag_values(8, 0);
    set_branch_address(tree_tag, tag_nhits, tag_values);

    Int_t rec_nhits;
    std::vector<Float_t> rec_values(8, 0);
    set_branch_address(tree_rec, rec_nhits, rec_values);

    Long64_t nentries = treeA->GetEntries();
    Long64_t nentries_tag = tree_tag->GetEntries();
    Long64_t tag_i = 0;
    Long64_t nentries_rec = tree_rec->GetEntries();
    Long64_t rec_i = 0;
    for (Long64_t i = 0; i < nentries; i++) {

        if (i % (nentries / 25) == 0) {
            cout << "Processing event " << i << " of " << nentries << endl;
        }

        auto fill_values = [](
                std::vector<std::vector<Double_t>> &values, Int_t nhits, std::vector<Float_t> track_values
        ) {
            values[0].push_back(track_values[0]);
            values[1].push_back(track_values[1]);
            values[2].push_back(track_values[2]);
            values[3].push_back(track_values[3]);
            values[4].push_back(track_values[4]);
            values[5].push_back(track_values[5]);
            values[6].push_back(track_values[6]);
            values[7].push_back(track_values[7]);
            values[8].push_back(static_cast<Double_t>(nhits));
        };


        treeA->GetEntry(i);
        gnn_tag_no_tracks = 0;
        gnn_rec_no_tracks = 0;

        tree_tag->GetEntry(tag_i);
        while (tag_evt == cur_evt && tag_run == cur_run) {
            fill_values(gnn_tag, tag_nhits, tag_values);
            gnn_tag_no_tracks++;
            tag_i++;
            if (tag_i >= nentries_tag) break;
            tree_tag->GetEntry(tag_i);
        }

        tree_rec->GetEntry(rec_i);
        while (rec_evt == cur_evt && rec_run == cur_run) {
            fill_values(gnn_rec, rec_nhits, rec_values);
            gnn_rec_no_tracks++;
            rec_i++;
            if (rec_i >= nentries_rec) break;
            tree_rec->GetEntry(rec_i);
        }


        // fill cloned tree
        treeOut->Fill();

        // clean
        for (auto &v : gnn_tag) v.clear();
        for (auto &v : gnn_rec) v.clear();
    }

    cout << "Writing to file" << endl;

    fileOut->cd();
    treeOut->Write();
    fileOut->Close();
    gnn_tag_file->Close();
    ana_file->Close();
}
