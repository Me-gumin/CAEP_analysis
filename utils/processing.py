import os
import pickle
import shutil
import tempfile
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
import mne
from scipy.stats import pearsonr
from scipy.signal import hilbert



#特征提取
def extract_all_features(epochs,ceap_band):
    """一次性提取所有特征, 缓存用"""
    data = epochs.get_data()  # (n_epochs, n_ch, n_times)
    chs = epochs.ch_names
    sfreq = epochs.info['sfreq']
    times = epochs.times * 1000
    n_ep, n_ch, n_t = data.shape
    evoked = data.mean(axis=0) * 1e6  # µV

    # ---- CAEP peaks (Cz) ----
    cz = chs.index('Cz') if 'Cz' in chs else 0
    cz_data = evoked[cz]
    features = {}
    for name, (t1, t2), find_max in [('P1',(30,80),True),('N1',(80,150),False),('P2',(150,250),True)]:
        mask = (times >= t1) & (times <= t2)
        idx = (np.argmax if find_max else np.argmin)(cz_data[mask]) + np.where(mask)[0][0]
        features[f'{name}_lat'] = times[idx]
        features[f'{name}_amp'] = cz_data[idx]
    features['N1P2'] = features['P2_amp'] - features['N1_amp']

    # ---- Band powers (Cz) ----
    freqs, psd = scipy_signal.welch(data[:, cz, :], fs=sfreq, nperseg=min(n_t, 128), axis=-1)
    psd_mean = 10 * np.log10(psd.mean(axis=0) + 1e-12)
    for band, (f1, f2) in ceap_band.items():
        mask = (freqs >= f1) & (freqs <= f2)
        features[f'power_{band}'] = np.mean(10 ** (psd_mean[mask] / 10))

    # ---- GFP (Global Field Power) ----
    gfp = np.std(evoked, axis=0)
    features['GFP_max'] = np.max(gfp[(times >= 30) & (times <= 250)])
    features['GFP_lat'] = times[30 + np.argmax(gfp[(times >= 30) & (times <= 250)])]

    # ---- ITC (Inter-Trial Coherence at Cz, alpha) ----

    alpha_data = mne.filter.filter_data(data[:, cz, :], sfreq, 8, 13, verbose=False)
    analytic = hilbert(alpha_data)
    itc = np.abs(np.mean(analytic / np.abs(analytic + 1e-12), axis=0))
    features['ITC_alpha_mean'] = np.mean(itc)

    # ---- SNR: signal power in P1-N1-P2 window vs baseline ----
    sig_mask = (times >= 30) & (times <= 250)
    base_mask = times < 0
    sig_pow = np.mean(evoked[cz, sig_mask] ** 2)
    base_pow = np.mean(evoked[cz, base_mask] ** 2) if base_mask.any() else 1e-12
    features['SNR_db'] = 10 * np.log10(sig_pow / (base_pow + 1e-12))

    # ---- Electrode importance (correlation of each ch with GFP) ----
    ch_importance = {}
    for i, ch in enumerate(chs):
        ch_importance[ch] = pearsonr(np.abs(evoked[i, (times>=30)&(times<=250)]),
                                      gfp[(times>=30)&(times<=250)])[0]
    features['channel_importance'] = ch_importance
    features['minimal_electrodes'] = sorted(ch_importance, key=ch_importance.get, reverse=True)[:6]

    return features, evoked, times, chs

def simulate_hearing_loss(epochs, snr_db):
    """对 epochs 加噪声模拟听损。返回带噪声的 evoked 对象"""
    data = epochs.get_data()  # (n_epochs, n_ch, n_times)
    sig_pow = np.mean(data ** 2)
    noise_pow = sig_pow / (10 ** (snr_db / 10))
    noise = np.random.RandomState(42).randn(*data.shape) * np.sqrt(noise_pow)
    noisy_data = data + noise

    # 用加噪数据创建新的 evoked
    info = epochs.info
    evoked = mne.EvokedArray(noisy_data.mean(axis=0), info, tmin=epochs.tmin)
    return evoked

def auto_preprocessing(DATA_DIR,SFREQ_TARGET,SPEECH_LABELS,PREPROC_DIR,
                       edf_file,vhdr_file,eeg_file,vmrk_file,events_file):
    try:
        os.makedirs(f"{DATA_DIR}/_up/eeg", exist_ok=True)
        # 判断格式
        if edf_file:
            # status.update(label="步骤 1/3: 读取 EDF/BDF 文件...")
            sub_name = edf_file.name.rsplit('.', 1)[0]
            tmp_edf = f"{DATA_DIR}/_up/{sub_name}.edf"
            with open(tmp_edf, 'wb') as f:
                f.write(edf_file.getbuffer())
            raw = mne.io.read_raw_edf(tmp_edf, preload=True, verbose=False)
        else:
            # status.update(label="步骤 1/3: 加载 BrainVision 数据...")
            sub_name = vhdr_file.name.split('_')[0]
            tmpdir = f"{DATA_DIR}/_up/eeg"
            with open(f"{tmpdir}/{sub_name}.vhdr", 'wb') as f:
                f.write(vhdr_file.getbuffer())
            with open(f"{tmpdir}/{sub_name}.eeg", 'wb') as f:
                f.write(eeg_file.getbuffer())
            if vmrk_file:
                with open(f"{tmpdir}/{sub_name}.vmrk", 'wb') as f: f.write(vmrk_file.getbuffer())
            if events_file:
                with open(f"{tmpdir}/{sub_name}_events.tsv", 'wb') as f: f.write(
                    events_file.getbuffer())
            raw = mne.io.read_raw_brainvision(f"{tmpdir}/{sub_name}.vhdr", preload=True, verbose=False)

        # 通用处理
        eeg_chs = [ch for ch in raw.ch_names
                   if raw.get_channel_types(picks=[ch])[0] == 'eeg'
                   and ch not in ('EP1', 'EP2', 'ECG', 'EOG')]
        if len(eeg_chs) < 2:
            eeg_chs = [ch for ch in raw.ch_names if ch not in ('EP1', 'EP2', 'ECG', 'EOG')]
        raw.pick(eeg_chs).resample(SFREQ_TARGET, verbose=False)
        raw.filter(0.5, 30, fir_design='firwin', verbose=False)

        # status.update(label="步骤 2/3: 提取事件...")
        # 优先用 events.tsv, 其次用 vmrk annotations, 最后用均匀分段
        tsv_p = f"{DATA_DIR}/_up/eeg/{sub_name}_events.tsv"
        if os.path.exists(tsv_p):
            df_ev = pd.read_csv(tsv_p, sep='\t')
        else:
            try:
                ev, _ = mne.events_from_annotations(raw, verbose=False)
                df_ev = pd.DataFrame({'onset': ev[:, 0] / raw.info['sfreq'], 'value': ev[:, 2]})
            except:
                df_ev = None

        if df_ev is not None and 'value' in df_ev.columns and df_ev['value'].between(7, 12).any():
            speech_df = df_ev[df_ev['value'].between(7, 12)]
        elif df_ev is not None and len(df_ev) > 0:
            speech_df = df_ev.head(40)
        else:
            # 等距生成伪事件
            n = min(40, int(raw.times[-1] / 2))
            speech_df = pd.DataFrame({'onset': np.linspace(1, raw.times[-1] - 1, n), 'value': [7] * n})

        eid_map = {}
        mne_ev = []
        for _, row in speech_df.iterrows():
            onset, val = float(row['onset']), int(row.get('value', 7))
            sample = int(round(onset * SFREQ_TARGET))
            name = SPEECH_LABELS.get(val, f'cond_{val}')
            if name not in eid_map: eid_map[name] = len(eid_map) + 1
            mne_ev.append([sample, 0, eid_map[name]])
        mne_ev = np.array(mne_ev)

        # status.update(label="步骤 3/3: 创建 Epochs & 保存...")
        epochs = mne.Epochs(raw, mne_ev, event_id=eid_map, tmin=-0.2, tmax=0.8,
                            baseline=(-0.2, 0), preload=True, reject=None, verbose=False)
        out = f"{PREPROC_DIR}/{sub_name}_caep_epochs-epo.fif"
        epochs.save(out, overwrite=True)
        from utils.load import load_summary
        sm = load_summary(PREPROC_DIR) or []
        sm.append({'sub_id': sub_name, 'n_epochs': len(epochs), 'n_channels': len(eeg_chs),
                   'ch_names': eeg_chs, 'sfreq': SFREQ_TARGET, 'speech_types': list(eid_map.keys())})
        with open(f"{PREPROC_DIR}/summary.pkl", 'wb') as f:
            pickle.dump(sm, f)
        return sub_name,epochs

    except Exception as e:
        print(f"预处理出错: {e}")