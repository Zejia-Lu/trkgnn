import ROOT

import pandas as pd


def load_ntuples(file_path: str, selections: list[str]):
    df_columns = ['bin', 'Truth', 'CKF', 'GNN']

    interests = {
        "Tag No_of_tracks": {
            "hist_setup": ("tag_no_tracks", f"tag_no_tracks", 20, 0, 20),
            "branches": ["TagTrk2_track_No", "TagTrk2_track_No", "gnn_tag_no_tracks"],
            "hists": [],
            "df": pd.DataFrame(columns=df_columns),
        },
        "Rec No_of_tracks": {
            "hist_setup": ("rec_no_tracks", f"rec_no_tracks", 20, 0, 20),
            "branches": ["RecTrk2_track_No", "RecTrk2_track_No", "gnn_rec_no_tracks"],
            "hists": [],
            "df": pd.DataFrame(columns=df_columns),
        }
    }

    rdf = ROOT.RDataFrame("dp", file_path)
    report = rdf.Report()
    for sel in selections:
        rdf = rdf.Filter(sel, sel)

    for name, info in interests.items():
        for branch in info["branches"]:
            info["hists"].append(rdf.Histo1D(info["hist_setup"], branch))

    rdf.Count().GetValue()

    # convert histogram to dataframe
    # for name, info in interests.items():

    a=0
    pass


if __name__ == '__main__':
    load_ntuples("/Users/avencast/PycharmProjects/trkgnn/workspace/test/merged.root", [""])
