import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import matplotlib.patches as mpatches
dummy_patch = mpatches.Patch(color='white', label='')
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import plotly.graph_objs as go

from starforge_mult_search.analysis.analyze_multiples_part2 import get_bound_snaps
from starforge_mult_search.analysis.figures.figure_preamble import *
import cgs_const as cgs

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

def plot_ecdf_plotly(x_vals, ys, errs=None, labels=None):
    traces = []
    for i, (x, y) in enumerate(zip(x_vals, ys)):
        trace = go.Scatter(
            x=x, y=y,
            mode='lines',
            name=labels[i] if labels else f"Series {i}",
            line=dict(width=2)
        )
        traces.append(trace)

        if errs is not None:
            trace_fill = go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate([y - errs[i], (y + errs[i])[::-1]]),
                fill='toself',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=False
            )
            traces.append(trace_fill)
    return traces

# def quad_plot(d_set1, extra_filt=None, type=0, unit=1, log=False):
def quad_plot(d_set1, extra_filt=None, type=0, unit=1, log=False):
    filt1 = (quasi_filter) & (final_bin_filter) & (bfb_filter)
    filt2 = (quasi_filter) & (final_bin_filter) & (~bfb_filter)
    if extra_filt is not None:
        filt1 = filt1 & extra_filt
        filt2 = filt2 & extra_filt

    d1 = d_set1[filt1][:, type] * unit
    d2 = d_set1[filt2][:, type] * unit
    if log:
        d1 = np.log10(d1)
        d2 = np.log10(d2)

    x_vals, ys, y_errs = get_ecdf_split((d1, d2), (seeds_lookup[filt1], seeds_lookup[filt2]),
                   (f"Final, BFB\n{len(d1)}", f"Final, Other\n{len(d2)}"),
                   levels=(10, 80), y_offset=-0.1, x_offset=[0, -0.2], ha=['left', 'right'])
    traces = plot_ecdf_plotly(x_vals, ys, errs=y_errs)
    return traces


##NOT normal survival filter -- for now just look at things that survive as binaries.
final_bin_filter = (my_data["final_bound_snaps_norm"]==1)
quasi_filter = my_data["quasi_filter"]
bfb_filter = my_data["same_sys_at_fst"].astype(bool)
pmult_before_bin = np.load("pmult_before_bin.npz")["pmult_filt"]
exchange_filter = ~pmult_before_bin

end_states = []

for row in my_data["bin_ids"]:
    bin_list = list(row)
    sys1_info = lookup_dict[bin_list[0]]
    sys2_info = lookup_dict[bin_list[1]]
    b1, b2, snaps = get_bound_snaps(sys1_info, sys2_info)

    end_states.append([b1[-1, LOOKUP_SMA], b1[-1, LOOKUP_ECC], min(b1[-1, LOOKUP_Q], b2[-1, LOOKUP_Q])])
end_states = np.array(end_states)
app = Dash()

# App layout
app.layout = [
    html.Div(children=''),
    html.Hr(),
    html.Div(["Input min mass:", dcc.Input(value="0", id='min_mass', type="number", style={'height':'35px', 'font-size': 30})]),
    dcc.Graph(figure={}, id='my-graph')
]

@callback(
    Output(component_id='my-graph', component_property='figure'),
    Input(component_id='min_mass', component_property='value')
)
def update_graph(min_mass):
    # traces = quad_plot(end_states, extra_filt=(my_data["mfinal_primary"] > min_mass), type=0, unit=cgs.pc / cgs.au, log=True)
    traces = quad_plot(end_states, extra_filt=(my_data["mfinal_primary"] > min_mass))

    return go.Figure(data=traces)

# Run the app
if __name__ == '__main__':
    app.run()