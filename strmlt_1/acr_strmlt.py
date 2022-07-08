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
@st.cache()
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

@st.cache()
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

@st.cache(allow_output_mutation=True)
def _combine_data_eeg(data, conds, dtype='bp'):
    return kup.combine_data_eeg(data, conds, dtype=dtype)

@st.cache(allow_output_mutation=True)
def _combine_data_lfp(data, conds, dtype='bp'):
    return kup.combine_data_lfp(data, conds, dtype=dtype)

@st.cache(allow_output_mutation=True)
def plot_raw_bp_single_chan(bp, chan):
    fig = px.line(bp.bp_melt().ch([chan]), x='datetime', y='Bandpower', facet_row='Band')
    return fig

@st.cache(allow_output_mutation=True)
def plot_a_band_multichan(bp, band):
    fig = px.line(bp, x='datetime', y=band, facet_row='channel')
    return fig

@st.cache(allow_output_mutation=True)
def plot_raw_bp_comp(df, band='delta'):
    fig = px.line(df, x='timedelta', y=band, facet_row='channel', color='Condition')
    return fig

# First, we are going to load all data needed for all plotting to come.
data, hyp, times = load_data(total_time=time_to_load)
hyp = load_dt_hypnos()
data = get_spg(data)
bp = get_bp(data, hyp)

st.title('ACR Analyses - Bandpower Plots')

#Then, the user can choose to plot some simple bandpower plots before moving to compare across conditions. 
st.markdown("## Simple Bandpower Plot Options")

plt_options = st.multiselect("Choose Desired Plots:", ['Full Bandpower Set', 'Single Bandpower, Multi-Chan'], default=None)
if plt_options: 
    expmt = st.selectbox("Choose Experiment:", sub_info['complete_key_list'])
    dstore = st.selectbox("Choose Data Store:", sub_info['stores'])
    key = expmt+'-e-bp' if dstore == 'EEGr' else expmt+'-f-bp'

if 'Full Bandpower Set' in plt_options:
    chan = st.selectbox("Choose Channel:", np.arange(1,16))
    fbp = plot_raw_bp_single_chan(bp[key], chan=1)
    st.plotly_chart(fbp)

if 'Single Bandpower, Multi-Chan' in plt_options:
    band = st.selectbox("Choose Band:", list(bp[key]))
    sbmc = plot_a_band_multichan(bp[key], band)
    st.plotly_chart(sbmc)


st.markdown("## Comparison Plots - Raw Bandpower")
show_rawbp = st.checkbox("Show Raw Bandpower Plots")
if show_rawbp:
    band = st.selectbox("Choose Band:", list(bd))
    dstore = st.selectbox("Choose Data Store:", sub_info['stores'])
    
    if dstore == 'EEGr':
        data = _combine_data_eeg(bp, conds=sub_info['complete_key_list'])
        
    elif dstore == 'LFP':
        data = _combine_data_lfp(bp, conds=sub_info['complete_key_list'])

    frbp = plot_raw_bp_comp(bp['concat'], band=band)
    st.plotly_chart(frbp)