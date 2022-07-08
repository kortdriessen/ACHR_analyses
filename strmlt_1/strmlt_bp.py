import kd_analysis.main.kd_utils as kd
import kd_analysis.main.kd_utils_pandas as kup
import pandas as pd
import tdt
import plotly.express as px
import plotly.io as pio
import kd_analysis.ACR.acr_utils as acu
import kd_analysis.ACR.acr_info as ai
import kd_analysis.main.kd_hypno as kh
import hypnogram as hp
import numpy as np
pio.templates.default = "plotly_dark"

import streamlit as st

# Information we need to load the data: 
bd = {}
bd['delta'] = slice(0.75, 4.1)
bd['theta'] = slice(4.1, 8.1)
bd['alpha'] = slice(8.1, 13.1)
bd['sigma'] = slice(11.1, 16.1)
bd['beta'] = slice(13.1, 30.1)
bd['gamma'] = slice(30.1, 100.1)

time_to_load=43200
times = ai.a9_times

sub_info = {}
sub_info['subject'] = 'ACR_9'
sub_info['complete_key_list'] = ['control1', 'laser1']
sub_info['stores'] = ['EEGr', 'LFP']



#Functions needed to load the data:
@st.cache(allow_output_mutation=True)
def load_data(total_time=None, hyp=False):
    sub = sub_info['subject']
    exp_list = sub_info['complete_key_list']
    paths = acu.get_paths(sub, sub_info['complete_key_list'])
    a = {}
    h={}
    
    for condition in exp_list:
        # This will need to be adjusted depending on which data stores are being used (EEGr, LFP, etc.)
        t1=times[condition]['bl_sleep_start']-30
        t2 = t1 + total_time if total_time else times[condition]['stim_off']
        print('starting load for', condition)
        a[condition+'-e-d'] = kup.tdt_to_pandas(paths[condition], t1=t1, t2=t2, channel=[1,2], store='EEGr')
        a[condition+'-f-d'] = kup.tdt_to_pandas(paths[condition], t1=t1, t2=t2, channel=[2,8,15], store='LFP_',)
        start_time = a[condition+'-e-d'].datetime.values[0]
        if hyp:
            h[condition] = acu.load_hypno_set(sub, condition, scoring_start_time=start_time)
        print('Done loading data for '+condition)
    return a, h, times

@st.cache(allow_output_mutation=True)
def get_spg(data, window_length=4, overlap=2):
    # Again, will need adjusting depending on which data stores are being used.
    for condition in sub_info['complete_key_list']:
        data[condition+'-e-s'] = kup.pd_spg(data[condition+'-e-d'], window_length=window_length, overlap=overlap)
        data[condition+'-f-s'] = kup.pd_spg(data[condition+'-f-d'], window_length=window_length, overlap=overlap)
    return data

@st.cache(allow_output_mutation=True)
def get_bp(data, hypno):
    bp = {}
    for condition in sub_info['complete_key_list']:
        bp[condition+'-e-bp'] = kup.pd_bp(data[condition+'-e-s'])
        bp[condition+'-e-bp'] = kup.add_states_to_data(bp[condition+'-e-bp'], hypno[condition])
        bp[condition+'-f-bp'] = kup.pd_bp(data[condition+'-f-s'])
        bp[condition+'-f-bp'] = kup.add_states_to_data(bp[condition+'-f-bp'], hypno[condition])
    return bp

@st.cache(allow_output_mutation=True)
def load_dt_hypnos(names=sub_info['complete_key_list']):
    dt_hypnos = {}
    sub = sub_info['subject']
    for name in names:
        path = '/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/'+sub+'/dt_hypnograms/hypno_'+name+'.txt'
        dt_hypnos[name] = hp.load_datetime_hypnogram(path)
    return dt_hypnos

# First, we are going to load the bandpower data needed for all plotting to come.
data, hyp, times = load_data(total_time=time_to_load)
hyp = load_dt_hypnos()
data = get_spg(data)
bp = get_bp(data, hyp)

st.title('ACR Analyses - Bandpower Plots')

#__________________________________________________________________________________
st.markdown("# Comparison Plots - Bandpower Relative to BL, State-Specific")
#___________________________________________________________________________________

@st.cache()
def bp_rel_2bl(bp_df, times_cond, state=['NREM']):
    start = bp_df.datetime.values[0]
    t1 = times_cond['stim_on_dt']
    t2 = times_cond['stim_off_dt']
    
    avg_period = slice(start, t1)
    bp = bp_df.filt_state(states=state)
    avg = bp.ts(avg_period)
    avg = avg.set_index(['datetime', 'channel'])
    avg = avg.groupby(level=['channel']).mean()
    bp = bp.set_index(['datetime', 'channel'])
    rel = bp/avg
    return rel.reset_index()

@st.cache()
def new_timedelta(df, interval=2):
    data_points = len(df)/len(pd.unique(df['channel']))
    total_time = data_points*interval
    td = np.arange(0, total_time, interval)
    new_td = np.repeat(td, len(pd.unique(df['channel'])))
    df['timedelta'] = new_td
    return df

@st.cache()
def smooth_df_col(df, col, period=2, smoothing_sigma=10):
    df = df.copy()
    fs = 1/period
    data = df[col]
    smoothed_data = kd.gaussian_smooth(data, sigma=smoothing_sigma, sampling_frequency=fs)
    df[col] = smoothed_data
    return df

@st.cache()
def comb_dataset(ds):
    keys = list(ds.keys())
    for k in keys:
        ds[k]['Condition'] = k
    ds['concat'] = (pd.concat(list(ds[k] for k in keys)))
    return ds['concat']

band = st.selectbox("Choose Band:", list(bd))
dtype = st.selectbox("Choose Data Type:", ['EEG', 'LFP'])

st.markdown('## EEG')
show_relbp_eeg = st.checkbox("Show Relative Bandpower Plots - EEG")
if show_relbp_eeg:
    rel_smooth_eeg = {}
    for exp in sub_info['complete_key_list']:
        key = exp+'-e-bp'
        rel_smooth_eeg[exp] = bp_rel_2bl(bp[key], times[exp])
        rel_smooth_eeg[exp] = new_timedelta(rel_smooth_eeg[exp])
        rel_smooth_eeg[exp] = smooth_df_col(rel_smooth_eeg[exp], band, period=2)
    eeg = comb_dataset(rel_smooth_eeg)
    eeg_fig = px.line(eeg, x='timedelta', y=band, color='Condition', facet_row='channel', title='Relative Bandpower - EEG')
    st.plotly_chart(eeg_fig)

st.markdown('## LFP')
show_relbp_lfp = st.checkbox("Show Relative Bandpower Plots - LFP")
if show_relbp_lfp:
    rel_smooth_lfp = {}
    for exp in sub_info['complete_key_list']:
        key = exp+'-f-bp'
        rel_smooth_lfp[exp] = bp_rel_2bl(bp[key], times[exp])
        rel_smooth_lfp[exp] = new_timedelta(rel_smooth_lfp[exp])
        rel_smooth_lfp[exp] = smooth_df_col(rel_smooth_lfp[exp], band, period=2)
    lfp = comb_dataset(rel_smooth_lfp)
    lfp_fig = px.line(lfp, x='timedelta', y=band, color='Condition', facet_row='channel', title='Relative Bandpower - LFP')
    st.plotly_chart(lfp_fig)