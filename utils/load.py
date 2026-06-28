import glob

import streamlit as st
import os
import mne

import pickle

# ==================== 数据加载 ====================
@st.cache_resource
def load_preprocessed(sub_id,PREPROC_DIR):
    path = f"{PREPROC_DIR}/{sub_id}_caep_epochs-epo.fif"
    if not os.path.exists(path):
        return None
    epochs = mne.read_epochs(path, preload=True, verbose=False)
    montage = mne.channels.make_standard_montage('standard_1020')
    epochs.set_montage(montage, on_missing='warn', verbose=False)
    return epochs

@st.cache_resource
def load_features(sub_id,ceap_bands,PREPROC_DIR):
    epochs = load_preprocessed(sub_id,PREPROC_DIR)
    if epochs is None:
        return None
    from utils.processing import extract_all_features
    return extract_all_features(epochs,ceap_bands)


@st.cache_data
def load_summary(PREPROC_DIR):
    path = f"{PREPROC_DIR}/summary.pkl"

    if not os.path.exists(path):
        return None

    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, list):
                return data
            return None
    except Exception as e:
        st.sidebar.warning(f"读取 summary.pkl 失败: {e}")
        return None

def fif_Load(PREPROC_DIR):
    fif_files = glob.glob(f"{PREPROC_DIR}/*_caep_epochs-epo.fif")
    new_sum = []
    for fp in fif_files:
        sid = os.path.basename(fp).replace('_caep_epochs-epo.fif', '')
        try:
            ep = mne.read_epochs(fp, preload=True, verbose=False)
            new_sum.append({'sub_id': sid, 'n_epochs': len(ep),
                           'n_channels': len(ep.ch_names), 'ch_names': ep.ch_names,
                           'sfreq': ep.info['sfreq'],
                           'speech_types': list(ep.event_id.keys())})
        except:
            pass
    return new_sum