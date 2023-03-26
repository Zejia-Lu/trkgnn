import pandas as pd

# local dependencies
from utility.Control import cfg
from utility.DataLoader import get_data_loaders
from utility.EverythingNeeded import get_item_from_dataloader, convert_batch_to_df


def get_event(input_dir, collection, index=0, e0=1, tree_name='dp', rndm=1):
    cfg['rndm'] = rndm
    cfg['data'] = {
        'collection': collection,
        'tree_name': tree_name,
        'E0': e0,
    }

    # load data
    data_loader = get_data_loaders(
        input_dir=input_dir,
        chunk_size=100,
        batch_size=1
    )
    loader, _ = next(data_loader)

    batch = get_item_from_dataloader(loader, index)
    df = convert_batch_to_df(batch)

    df_edge_value = pd.concat([df['edge'], df['y']], axis=1)
    df_edge_value['predict'] = df_edge_value['truth']
    merged_df = df_edge_value.merge(df['node'], left_on='start', right_index=True)

    df_edge = merged_df.merge(df['node'], left_on='end', right_index=True, suffixes=('_start', '_end'))
    df_node = df['node']

    return df_node, df_edge


if __name__ == '__main__':
    r = get_event(
        input_dir='/Users/avencast/PycharmProjects/trkgnn/workspace/bepc.magnet/Tracker_GNN.root',
        collection='TagTrk1',
        index=0
    )

    print(r)
