import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import matplotlib.patches as mpatches
from starforge_mult_search.analysis.analyze_stack import subtract_path

dummy_patch = mpatches.Patch(color='white', label='')
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import plotly.graph_objs as go

from starforge_mult_search.analysis.analyze_multiples_part2 import get_bound_snaps
from starforge_mult_search.analysis.figures.figure_preamble import *
import cgs_const as cgs

import seaborn as sns
colors_hex = sns.color_palette("colorblind").as_hex()

def get_spin_angle(tmp_spin1, tmp_spin2):
    return np.dot(tmp_spin1, tmp_spin2) / np.linalg.norm(tmp_spin1) / np.linalg.norm(tmp_spin2)

def spin_angle_from_lookup(spin_lookup, tmp_id1, tmp_id2, sel_bin_snap):
    tmp_spin1 = spin_lookup[str(tmp_id1)][sel_bin_snap][1:]
    tmp_spin2 = spin_lookup[str(tmp_id2)][sel_bin_snap][1:]
    ##Spin vectors
    spin_misalign = get_spin_angle(tmp_spin1, tmp_spin2)
    return spin_misalign

##SEPARATE OUT PLOTTING CODE(!!)
def ecdf(data):
    sorted_data = np.sort(data)
    y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return sorted_data, y

def get_ecdf_split(datasets, datasets_seeds, labels, **kwargs):
    ecdf_mean = []
    ecdf_std = []
    x_vals = []
    for dd, ss in zip(datasets, datasets_seeds):
        x_vals.append(
            np.linspace(np.min(dd[(~np.isinf(dd)) & (~np.isnan(dd))]), np.max(dd[(~np.isinf(dd)) & (~np.isnan(dd))]),
                        500))
        tmp_ecdfs = []
        su = np.unique(ss)
        for tmp_seed in su:
            tmp_data = dd[ss == tmp_seed]
            tmp_x, tmp_y = ecdf(tmp_data[(~np.isinf(tmp_data)) & (~np.isnan(tmp_data))])
            ecdf_y_interp = np.interp(x_vals[-1], tmp_x, tmp_y)  # Interpolate ECDF to common x-values
            tmp_ecdfs.append(ecdf_y_interp)
        ecdf_mean.append(np.mean(tmp_ecdfs, axis=0).ravel())
        ecdf_std.append(np.std(tmp_ecdfs, axis=0).ravel())
    # annotate_multiple_ecdf_err(x_vals, ecdf_mean, ecdf_std, labels, **kwargs)
    return x_vals, ecdf_mean, ecdf_std

def plot_ecdf_plotly(x_vals, ys, errs=None, labels=None, colors=None):
    traces = []
    for i, (x, y) in enumerate(zip(x_vals, ys)):
        color=None
        if colors is not None:
            color = colors[i]
        trace = go.Scatter(
            x=x, y=y,
            mode='lines',
            name=labels[i] if labels else f"Series {i}",
            line=dict(width=2, color=colors[i]),
        )
        traces.append(trace)

        if errs is not None:
            trace_fill = go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate([y - errs[i], (y + errs[i])[::-1]]),
                fill='toself',
                fillcolor=color if color else 'rgba(0,100,80)',
                opacity=0.2,
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=False
            )
            traces.append(trace_fill)
    return traces

# def quad_plot(d_set1, extra_filt=None, type=0, unit=1, log=False):
def quad_plot(d_set1, filt1, filt2, extra_filt=None, type=0, unit=1, log=False):
    if extra_filt is not None:
        filt1 = filt1 & extra_filt
        filt2 = filt2 & extra_filt

    d1 = d_set1.loc[filt1][type].to_numpy() * unit
    d2 = d_set1.loc[filt2][type].to_numpy() * unit
    if log:
        d1 = np.log10(d1)
        d2 = np.log10(d2)

    KS = ks_2samp(d1, d2).pvalue
    labels = (f"Final, BFB\n{len(d1)}", f"Final, Other\n{len(d2)}" +"\n"+ f"KS: {KS:.2g}")
    x_vals, ys, y_errs = get_ecdf_split((d1, d2), (d_set1.loc[filt1]["seeds_lookup"], d_set1.loc[filt2]["seeds_lookup"]),
                   labels,
                   levels=(10, 80), y_offset=-0.1, x_offset=[0, -0.2], ha=['left', 'right'])
    traces = plot_ecdf_plotly(x_vals, ys, errs=y_errs, labels=labels, colors=(colors_hex[0], colors_hex[1]))
    return traces

def make_filter_dropdown(filter_id, label):
    return html.Div([
        html.Label(f"{label}:"),
        dcc.Dropdown(
            id=filter_id,
            options=[
                {'label': 'All', 'value': 'all'},
                {'label': 'True', 'value': 'true'},
                {'label': 'False', 'value': 'false'}
            ],
            value='all',
            clearable=False,
            style={'width': '100px'}
        )
    ])

def apply_trinary_filter(condition, selection):
    if selection == 'all':
        return np.ones(len(condition)).astype(bool)
    elif selection == 'true':
        return condition
    elif selection == 'false':
        return ~condition

def get_soft_time(id1, id2):
    delta = subtract_path(path_lookup[f"{id1}"][:, pxcol:pzcol + 1], path_lookup[f"{id2}"][:, pxcol:pzcol + 1])
    delta = np.sum(delta * delta, axis=1)**.5 * cgs.pc / cgs.au
    idx_soft = np.where(delta < 40)[0]
    if len(idx_soft) == 0:
        return np.inf
    else:
        return idx_soft[0]

##NOT normal survival filter -- for now just look at things that survive as binaries.
final_bin_filter = (my_data["final_bound_snaps_norm"]==1)
quasi_filter = my_data["quasi_filter"]
bfb_filter = my_data["same_sys_at_fst"].astype(bool)
pmult_before_bin = np.load("pmult_before_bin.npz")["pmult_filt"]
exchange_filter = ~pmult_before_bin

bprops = []
my_data["soft_time"] = np.zeros(len(my_data["bin_ids"]))
for ii, row in enumerate(my_data["bin_ids"]):
    bin_list = list(row)
    id1 = bin_list[0]
    id2 = bin_list[1]
    sys1_info = lookup_dict[id1]
    sys2_info = lookup_dict[id2]
    ##Snaps are all the snapshots the stars are in the same system -- not the snapshots they are in the same binary.
    b1, b2, snaps = get_bound_snaps(sys1_info, sys2_info)
    sel_bin_snap = int(my_data["final_bound_snaps"][ii])
    spin_ang = [spin_angle_from_lookup(spin_lookup, id1, id2, ss) for ss in b1[:, LOOKUP_SNAP].astype(int)]
    sim_end_snap = lookup_dict[id1][0, -1]
    ##Softening filter??
    ##get_first_soft_time--can be done separately...
    q_no_halo =  np.min((b1[:, LOOKUP_M] / b2[:, LOOKUP_M], b2[:, LOOKUP_M] / b1[:, LOOKUP_M]), axis=0)
    my_data["soft_time"][ii] = get_soft_time(id1, id2)
    ##Also have toggle for halo masses(!)
    ## Mass ratio without halos -- controlled by button. Need new column or can be reconstructed from masses...
    # bprops.append([b1[:, LOOKUP_SMA], b1[:, LOOKUP_ECC], np.min((b1[:, LOOKUP_Q], b2[:, LOOKUP_Q]), axis=0),
    #                b1[:, LOOKUP_SNAP] - b1[0,LOOKUP_SNAP], b1[:, LOOKUP_SNAP] / sim_end_snap, b1[:, LOOKUP_SNAP], np.ones(len(b1)) * ii, spin_ang,
    #                np.min((b1[:, LOOKUP_M] / b2[:, LOOKUP_M], b2[:, LOOKUP_M] / b1[:, LOOKUP_M]), axis=0)])
    bprops.append([b1[:, LOOKUP_SMA], b1[:, LOOKUP_ECC], np.min((b1[:, LOOKUP_Q], b2[:, LOOKUP_Q]), axis=0),
                   b1[:, LOOKUP_SNAP] - b1[0,LOOKUP_SNAP], b1[:, LOOKUP_SNAP] / sim_end_snap,  np.ones(len(b1)) * ii, b1[:, LOOKUP_SNAP], spin_ang, q_no_halo])

import pandas as pd
bprops = pd.concat([pd.DataFrame(row).T for row in bprops], ignore_index=0)
bprops.rename(columns={0:"sma", 1:"e", 2:"q", 3:"delay", 4:"snap_norm", 5: "bin_idx", 6:"snap", 7:"spin_ang", 8:"q_no_halo"}, inplace=True)
bprops.set_index("bin_idx", inplace=True)

qfilter = pd.DataFrame(quasi_filter)
qfilter.index.name = "bin_idx"
qfilter.rename(columns={0:"quasi_filter"}, inplace=True)

efilter = pd.DataFrame(exchange_filter)
efilter.index.name = "bin_idx"
efilter.rename(columns={0:"exchange_filter"}, inplace=True)

bfilter = pd.DataFrame(bfb_filter)
bfilter.index.name = "bin_idx"
bfilter.rename(columns={0:"bfb_filter"}, inplace=True)

mfilter = pd.DataFrame(my_data["mfinal_primary"])
mfilter.index.name = "bin_idx"
mfilter.rename(columns={0:"mfinal_primary"}, inplace=True)

##TO DO: ADD THIS FILTER
soft_time = pd.DataFrame(my_data["soft_time"])
soft_time.index.name = "bin_idx"
soft_time.rename(columns={0:"soft_time"}, inplace=True)

seeds_lookup = pd.DataFrame(seeds_lookup)
seeds_lookup.index.name = "bin_idx"
seeds_lookup.rename(columns={0:"seeds_lookup"}, inplace=True)


bprops = pd.merge(bprops, qfilter, on="bin_idx", how="outer")
bprops = pd.merge(bprops, efilter, on="bin_idx", how="outer")
bprops = pd.merge(bprops, bfilter, on="bin_idx", how="outer")
bprops = pd.merge(bprops, mfilter, on="bin_idx", how="outer")
bprops = pd.merge(bprops, soft_time, on="bin_idx", how="outer")
bprops = pd.merge(bprops, seeds_lookup, on="bin_idx", how="outer")

# bprops = np.array(bprops)
app = Dash(__name__, external_scripts=['https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML'])
# mathjax = 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML'
# app.scripts.append_script({ 'external_url' : mathjax })

# App layout
app.layout = [
    html.Div(children='\( \gamma \)'),
    html.P(children='Delicious \(\pi\) is inline with my goals.'),
    html.Hr(),
    html.Div(["Input min mass:", dcc.Input(value="0", id='min_mass', type="number", style={'height':'35px', 'font-size': 30})]),
    html.Div([
        "Time to plot (from formation; final=plot properties of final binaries):",
        dcc.Input(value="final", id='snap_from_form', style={'height': '35px', 'font-size': 30})
    ]),
    dcc.RadioItems(
        id='halo_toggle',
        options=[
            {'label': 'Plot q with halo', 'value': True},
            {'label': 'Plot q with no halo', 'value': False}
        ],
        value=True,  # default value to True (Column 1)
        labelStyle={'display': 'inline-block'}
    ),
    dcc.RadioItems(
        id='soft_toggle',
        options=[
            {'label': 'Remove bins affected by softening from top plots', 'value': True},
            {'label': 'Keep bins affected by softening from top plots', 'value': False}
        ],
        value=True,  # default value to True (Column 1)
        labelStyle={'display': 'inline-block'}
    ),
    html.Div([
        html.H4("Dataset A Filters"),
        make_filter_dropdown('a_ex', 'Exchange Filter'),
        make_filter_dropdown('a_bfb', 'BFB Filter'),
    ], style={'display': 'inline-block', 'margin': '20px'}),

    html.Div([
        html.H4("Dataset B Filters"),
        make_filter_dropdown('b_ex', 'Exchange Filter'),
        make_filter_dropdown('b_bfb', 'BFB Filter'),
    ], style={'display': 'inline-block', 'margin': '20px'}),
    html.Div([
        dcc.Graph(id='graph1'),
        dcc.Graph(id='graph2'),
        dcc.Graph(id='graph3'),
        dcc.Graph(id='graph4'),
    ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '20px'}),
]

@callback(
    Output('graph1', 'figure'),
    Output('graph2', 'figure'),
    Output('graph3', 'figure'),
    Output('graph4', 'figure'),
    Input(component_id='min_mass', component_property='value'),
    Input('a_ex', 'value'),
    Input('a_bfb', 'value'),
    Input('b_ex', 'value'),
    Input('b_bfb', 'value'),
    Input('snap_from_form', 'value'),
    Input('halo_toggle', 'value'),
    Input('soft_toggle', 'value')
)
def update_graph(min_mass, a_ex, a_bfb, b_ex, b_bfb, snap_from_form, halo_toggle, soft_toggle):
    # extra_filt = (my_data["quasi_filter"]) & (my_data["mfinal_primary"] > float(min_mass))
    if snap_from_form=="final":
        bprops_filt = bprops.loc[bprops["snap_norm"]==1]
    else:
        bprops_filt = bprops.loc[bprops["delay"]==float(snap_from_form)]

    extra_filt = bprops_filt["quasi_filter"] & (bprops_filt["mfinal_primary"] > float(min_mass))
    filt_a1 = apply_trinary_filter(bprops_filt["exchange_filter"], a_ex)
    filt_a2 = apply_trinary_filter(bprops_filt["bfb_filter"], a_bfb)

    filt_b1 = apply_trinary_filter(bprops_filt["exchange_filter"], b_ex)
    filt_b2 = apply_trinary_filter(bprops_filt["bfb_filter"], b_bfb)

    qtype = "q"
    if not halo_toggle:
        qtype = "q_no_halo"  # Use 'col2' if False
    extra_filt_mod = extra_filt & (bprops_filt["snap"] < bprops_filt["soft_time"])
    if not soft_toggle:
        extra_filt_mod = extra_filt

    ##Add soft filter to traces1 and traces2
    ##by modifying the extra_filt(!) -- Need to get soft filter online first.
    ##extra_filt = extra_filt & (bprops["snap"] < bprops["soft"])
    traces1 = quad_plot(bprops_filt, filt_a1 & filt_a2, filt_b1 & filt_b2, extra_filt=(extra_filt_mod), type="sma", unit=cgs.pc / cgs.au, log=True)
    traces2 = quad_plot(bprops_filt, filt_a1 & filt_a2, filt_b1 & filt_b2, extra_filt=(extra_filt_mod), type="e")
    traces3 = quad_plot(bprops_filt, filt_a1 & filt_a2, filt_b1 & filt_b2, extra_filt=(extra_filt), type=qtype)
    traces4 = quad_plot(bprops_filt, filt_a1 & filt_a2, filt_b1 & filt_b2, extra_filt=(extra_filt), type="spin_ang")

    f1, f2, f3, f4 = go.Figure(data=traces1), go.Figure(data=traces2), go.Figure(data=traces3), go.Figure(data=traces4)
    f1.update_layout(xaxis_title="log(a [au])", yaxis_title="CDF")
    f2.update_layout(xaxis_title="e", yaxis_title="CDF")
    f3.update_layout(xaxis_title="q", yaxis_title="CDF")
    f4.update_layout(xaxis_title="cos(ϕₛ)", yaxis_title="CDF")

    return (f1, f2, f3, f4)

# Run the app
if __name__ == '__main__':
    app.run()