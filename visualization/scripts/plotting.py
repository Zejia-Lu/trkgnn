import os
import datetime

import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

color_dict = {
    'Correct': '#965601',
    'False Negative': '#da4892',
    'False Positive': '#1c965c',
}
line_dict = {
    'Correct': 'solid',
    'False Negative': 'dash',
    'False Positive': 'solid',
}


def select_df_with_cut(df_in, cut_value):
    def classify(a, b, c):
        if a > 0.5 and b >= c:
            return 'Correct'
        elif a > 0.5 and b < c:
            return 'False Negative'
        elif a < 0.5 and b >= c:
            return 'False Positive'
        else:
            return 'ignore'

    df_in['category'] = df_in.apply(lambda row: classify(row['truth'], row['predict'], cut_value), axis=1)
    df_in = df_in.loc[df_in['category'] != 'ignore']
    return df_in


def plot_xyz_plotly(node, edge, threshold=0.5, no_predict=False):
    df_edges = select_df_with_cut(edge, threshold)
    cat_dict = {k: 0 for k in edge['category'].unique()}
    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.0
    )

    axis_attr = dict(
        showgrid=False,
        mirror=True,
        linecolor="#666666", gridcolor='#d9d9d9',
        zeroline=False,
    )

    for (x, y, xi) in zip(['x', 'y'], ['z', 'z'], [1, 2]):
        # Add scatter plot for node positions
        fig.add_trace(
            go.Scatter(
                x=node[x],
                y=node['z'],
                mode='markers',
                marker=dict(color='#965601', size=7),
                # name='Nodes',
                showlegend=False,
            ), row=1, col=xi,
        )
        for i in range(len(df_edges)):
            edge = df_edges.iloc[i]
            if no_predict and edge['category'] != "Correct": continue
            fig.add_trace(
                go.Scatter(
                    x=[edge[f'{x}_start'], edge[f'{x}_end']],
                    y=[edge[f'{y}_start'], edge[f'{y}_end']],
                    mode='lines',
                    line=dict(
                        width=1 if edge['category'] == "Correct" else 2,
                        color=color_dict[edge['category']],
                        dash=line_dict[edge['category']],
                    ),
                    legendgroup=edge['category'],
                    name=edge['category'] if cat_dict[edge['category']] == 0 else None,
                    showlegend=False if no_predict else (not cat_dict[edge['category']]),
                ), row=1, col=xi,
            )

            cat_dict[edge['category']] = 1

        fig.update_xaxes(title_text=f'{x} [mm]', row=1, col=xi, range=[-275, 275], **axis_attr)
        fig.update_yaxes(title_text=f'{y} [mm]' if xi == 1 else "", row=1, col=xi, **axis_attr)

    fig.update_layout(
        width=1200,
        height=700,
        autosize=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=0.99,
            # font=dict(size=16),
            traceorder='reversed',
        ),
    )

    fig.show()


def plot_xyz_plotly_3d(node, edge, threshold=0.5, no_predict=True):
    fig = go.Figure()

    # Add scatter plot for node positions
    fig.add_trace(
        go.Scatter3d(
            x=node['x'],
            y=node['y'],
            z=node['z'],
            mode='markers',
            marker=dict(color='blue', size=5),
            name='Nodes',
            showlegend=False
        )
    )

    df_edges = select_df_with_cut(edge, threshold)
    for i in range(len(df_edges)):
        edge = df_edges.iloc[i]
        if no_predict and edge['category'] != "Correct": continue
        fig.add_trace(
            go.Scatter3d(
                mode='lines',
                x=[edge['x_start'], edge['x_end']],
                y=[edge['y_start'], edge['y_end']],
                z=[edge['z_start'], edge['z_end']],
                line=dict(
                    width=1,
                    color=color_dict[edge['category']],
                    dash=line_dict[edge['category']],
                ),
                showlegend=False
            )
        )

    # Set the background color and grid color
    fig.update_layout(
        scene=dict(
            bgcolor='black',
            xaxis=dict(gridcolor='lightgrey', zerolinecolor='lightgrey', showspikes=False, range=[-275, 275]),
            yaxis=dict(gridcolor='lightgrey', zerolinecolor='lightgrey', showspikes=False, range=[-275, 275]),
            zaxis=dict(gridcolor='lightgrey', zerolinecolor='lightgrey', showspikes=False),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
    )

    # Set the camera position and orientation
    camera = dict(
        eye=dict(x=1, y=1, z=-2),  # Position the camera on the -z axis
        up=dict(x=0, y=1, z=0),  # Set the "up" direction to be along the +y axis
        center=dict(x=0, y=0, z=0)  # Set the center of the scene
    )
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            camera=camera,
        )
    )
    fig.update_layout(width=1000, height=1000)

    fig.show()


def read_local_csv(local_csv_file):
    if os.path.exists(local_csv_file) is False:
        raise FileNotFoundError(f"File {local_csv_file} not found")

    """Read the local CSV file and return the dataframe and the timestamp"""
    timestamp = os.path.getmtime(local_csv_file)
    dt_object = datetime.datetime.fromtimestamp(timestamp)
    # Parse and display the CSV data
    df = pd.read_csv(local_csv_file)
    return df, f"{dt_object.year} - {dt_object.month} - {dt_object.day}   {dt_object.hour} : {dt_object.minute}"


def visual_summary_log(df, t):
    fig = go.Figure()

    y_max, y_min = df.max()[['train_loss', 'valid_loss']].max(), df.min()[['train_loss', 'valid_loss']].min()

    df_new = df[['epoch', 'train_loss', 'valid_loss', 'valid_acc']]. \
        groupby('epoch').transform("mean").drop_duplicates(keep='last', subset=['train_loss'])

    fig.data = []

    fig.add_trace(
        go.Scatter(
            x=df.index, y=df['train_loss'], mode='lines', name="itr: train", legendgroup="Itr Loss",
            legendgrouptitle_text="Itr Loss", line=dict(dash='dot'))
    )
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df['valid_loss'], mode='lines', name="itr: valid", legendgroup="Itr Loss",
            line=dict(dash='dot')
        ))
    fig.add_trace(
        go.Scatter(
            x=df_new.index,
            y=-np.log10(1 - df_new['valid_acc']),
            # y=df['valid_acc'],
            mode='lines', name="itr: accuracy",
            line=dict(dash='dot', color="#11ADF0"), yaxis='y2',
            legendgroup="Edge Label",
            legendgrouptitle_text="Edge Label",
        )
    )

    fig.add_trace(go.Scatter(
        x=df_new.index, y=df_new['train_loss'], mode='lines+markers', name="epoch: train",
        legendgroup="Epoch Loss", legendgrouptitle_text="Epoch Loss"
    ))
    fig.add_trace(go.Scatter(
        x=df_new.index, y=df_new['valid_loss'], mode='lines+markers', name="epoch: valid",
        legendgroup="Epoch Loss"
    ))

    for lr in df.drop_duplicates(keep="first", subset=['lr'])['lr'].items():
        print(lr)
        fig.add_vline(x=lr[0], line_width=2, line_dash="dash", line_color="grey")
        fig.add_annotation(
            text=f'$\eta = {lr[1]}$',
            x=lr[0] + 0.5, y=np.log10(y_min) * 1.02,  # Set the position using numeric values
            xanchor='left',
            # yanchor='bottom',
            showarrow=False,
            font=dict(size=14, color='grey')
        )

    fig.update_layout(
        title=f"{t} --> Epoch: {df['epoch'].iloc[-1]}, Iteration: {df['itr'].iloc[-1]}",
        width=1200,
        height=700,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title_text="Total iteration",
            showgrid=False,
            mirror=True,
            showline=True,
            zeroline=False,
            linewidth=2,
            linecolor='#666666', gridcolor='#d9d9d9',
            domain=[0.1, 0.85],
        ),
        yaxis=dict(
            title=r"Loss",
            titlefont=dict(color="#d62728"),
            tickfont=dict(color="#d62728"),
            showgrid=False,
            linecolor="#d62728", gridcolor='#d9d9d9',
            zeroline=False,
            type="log",
            range=[np.log10(y_min) * 1.05, np.log10(y_max) * 1.05],
        ),
        yaxis2=dict(
            # title=r"$\textrm{95% CI }{\Large \kappa_{2V}}\textrm{ Interval}$",
            title=r"$-log(\epsilon_{error})$",
            titlefont=dict(color="#11ADF0"),
            tickfont=dict(color="#11ADF0"),
            # anchor="free",
            # overlaying="y",
            side="right",
            # position=0.0,
            showgrid=False,
            linecolor="#11ADF0", gridcolor='#d9d9d9',
            zeroline=False,
            # type="log",
            # range=[3.0, 3.7],
        ),
    )

    return fig
