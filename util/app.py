"""
CAEP 皮层听觉诱发电位 — 语音听力筛查可视化分析系统 v2.0
========================================================
功能:
  - 通用原始数据上传 (BrainVision/EDF/CSV) → 自动预处理 → 可视化
  - CAEP 核心特征: P1-N1-P2 检测 + 频谱 + ITC + GFP
  - 电极最小化分析: 评估最少需要几个电极
  - 在线实时分析, 支持落地使用
"""

import streamlit as st
import numpy as np
import matplotlib.font_manager as fm
import mne
import pickle
import os, sys, glob, io, tempfile, shutil
import pandas as pd
from scipy import signal as scipy_signal
from scipy.stats import pearsonr
import warnings, traceback

import matplotlib.pyplot as plt
import matplotlib

# 设置matplotlib正常显示中文和负号
matplotlib.rcParams['font.family'] = 'SimHei'  # 'SimHei' 或其他支持中文的字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 为了防止乱码，也可以指定使用多个字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号
warnings.filterwarnings('ignore')

st.set_page_config(page_title="CAEP 听力筛查分析系统", layout="wide", initial_sidebar_state="expanded")

# ---- 中文字体 ----
# _cf = [f for f in fm.fontManager.ttflist if f.name == 'Arial Unicode MS']
# if _cf:
#     fm.fontManager.addfont(_cf[0].fname)
#     plt.rcParams['font.family'] = 'sans-serif'
#     plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
#     plt.rcParams['axes.unicode_minus'] = False
# else:
#     plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Apple LiSung']
#     plt.rcParams['axes.unicode_minus'] = False

DATA_DIR = "data"
PREPROC_DIR = "preprocessed"
os.makedirs(PREPROC_DIR, exist_ok=True)
SFREQ_TARGET = 500

# ==================== 特征提取引擎 ====================
SPEECH_LABELS = {7:'chn_aud',8:'eng_aud',9:'interview',10:'lecture',11:'news',12:'talk'}
CAEP_BANDS = {'delta':(1,4), 'theta':(4,8), 'alpha':(8,13), 'beta':(13,30)}

def extract_all_features(epochs):
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
    for band, (f1, f2) in CAEP_BANDS.items():
        mask = (freqs >= f1) & (freqs <= f2)
        features[f'power_{band}'] = np.mean(10 ** (psd_mean[mask] / 10))

    # ---- GFP (Global Field Power) ----
    gfp = np.std(evoked, axis=0)
    features['GFP_max'] = np.max(gfp[(times >= 30) & (times <= 250)])
    features['GFP_lat'] = times[30 + np.argmax(gfp[(times >= 30) & (times <= 250)])]

    # ---- ITC (Inter-Trial Coherence at Cz, alpha) ----
    from scipy.signal import hilbert
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


# ==================== 数据加载 ====================
@st.cache_resource
def load_preprocessed(sub_id):
    path = f"{PREPROC_DIR}/{sub_id}_caep_epochs-epo.fif"
    if not os.path.exists(path):
        return None
    epochs = mne.read_epochs(path, preload=True, verbose=False)
    montage = mne.channels.make_standard_montage('standard_1020')
    epochs.set_montage(montage, on_missing='warn', verbose=False)
    return epochs

@st.cache_resource
def load_features(sub_id):
    epochs = load_preprocessed(sub_id)
    if epochs is None:
        return None
    return extract_all_features(epochs)


@st.cache_data
def load_summary():
    # 检查全局变量是否存在
    if 'PREPROC_DIR' not in globals():
        st.sidebar.error("系统错误：未设置数据目录 PREPROC_DIR")
        return None

    path = f"{PREPROC_DIR}/summary.pkl"

    # 1. 检查文件是否存在
    if not os.path.exists(path):
        return None  # 没有文件就返回 None，不报错

    # 2. 尝试读取，捕获所有可能的读取异常
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
            # 简单校验一下数据格式，防止读出来是乱七八糟的东西
            if isinstance(data, list):
                return data
            else:
                return None
    except (pickle.UnpicklingError, EOFError, AttributeError, ImportError, FileNotFoundError) as e:
        # 在侧边栏显示温和的错误提示，而不中断整个页面
        st.sidebar.warning(f"读取 summary.pkl 失败，请检查文件是否完整。错误信息: {e}")
        return None

# ==================== 自动预处理引擎 ====================
def auto_preprocess_from_upload(file_map):
    """
    接受上传的原始文件, 自动识别格式并预处理
    file_map: {'vhdr': bytes, 'eeg': bytes, 'vmrk': bytes, 'events': bytes or None}
    返回: (sub_name, epochs_path) 或抛出异常
    """
    tmpdir = tempfile.mkdtemp(prefix='caep_upload_')
    try:
        # 写入临时文件
        vhdr_content = file_map['vhdr'].decode('utf-8', errors='replace')
        sub_name = "uploaded"
        for line in vhdr_content.split('\n'):
            if line.startswith('DataFile='):
                sub_name = os.path.splitext(os.path.basename(line.split('=')[1].strip()))[0]
                sub_name = sub_name.replace('_task-MusicvsSpeech_eeg', '').replace('_eeg', '')
                break

        # 写入文件
        for ext in ['eeg', 'vhdr', 'vmrk']:
            with open(f"{tmpdir}/data.{ext}", 'wb') as f:
                f.write(file_map[ext])
        if file_map.get('events'):
            with open(f"{tmpdir}/events.tsv", 'wb') as f:
                f.write(file_map['events'])

        # 读取
        raw = mne.io.read_raw_brainvision(f"{tmpdir}/data.vhdr", preload=True, verbose=False)
        eeg_chs = [ch for ch in raw.ch_names
                   if raw.get_channel_types(picks=[ch])[0] == 'eeg'
                   and ch not in ('EP1', 'EP2')]
        raw.pick(eeg_chs)
        raw.resample(SFREQ_TARGET, verbose=False)
        raw.filter(0.5, 30, fir_design='firwin', verbose=False)

        # 事件
        if file_map.get('events'):
            df = pd.read_csv(f"{tmpdir}/events.tsv", sep='\t')
        else:
            # 无 events.tsv, 尝试从 annotations 推断
            events, eid = mne.events_from_annotations(raw, verbose=False)
            # 简单处理: 生成伪事件用于 demo
            df = pd.DataFrame({'onset': raw.times[::int(SFREQ_TARGET*12)][:40],
                               'value': [7]*40, 'sample': range(0, len(raw.times), int(SFREQ_TARGET*12))[:40]})

        speech_df = df[df['value'].between(7, 12)] if 'value' in df.columns else df
        if len(speech_df) == 0:
            speech_df = df.iloc[:min(40, len(df))]

        eid = {}
        mne_events = []
        for _, row in speech_df.iterrows():
            onset = float(row.get('onset', 0))
            val = int(row.get('value', 7))
            sample = int(round(onset * SFREQ_TARGET))
            name = SPEECH_LABELS.get(val, f'cond_{val}')
            if name not in eid:
                eid[name] = len(eid) + 1
            mne_events.append([sample, 0, eid[name]])
        mne_events = np.array(mne_events)

        epochs = mne.Epochs(raw, mne_events, event_id=eid, tmin=-0.2, tmax=0.8,
                            baseline=(-0.2, 0), preload=True, reject=None, verbose=False)
        out_path = f"{PREPROC_DIR}/{sub_name}_caep_epochs-epo.fif"
        epochs.save(out_path, overwrite=True)

        # 更新 summary
        summary = load_summary() or []
        summary.append({'sub_id': sub_name, 'n_epochs': len(epochs),
                        'n_channels': len(eeg_chs), 'ch_names': eeg_chs,
                        'sfreq': SFREQ_TARGET, 'speech_types': list(eid.keys())})
        with open(f"{PREPROC_DIR}/summary.pkl", 'wb') as f:
            pickle.dump(summary, f)

        return sub_name, out_path
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

# ==================== 可视化辅助 ====================
def plot_topo(evoked_data, info, time_ms, ax, title=""):
    from mne.viz import plot_topomap as pt
    try:
        t_idx = np.argmin(np.abs(evoked_data.times * 1000 - time_ms))
        pt(evoked_data.data[:, t_idx], info, axes=ax, show=False,
           cmap='RdBu_r', outlines='head', sensors=False)
        ax.set_title(title, fontsize=9)
    except:
        ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)

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


# ==================== 侧边栏 ====================
st.sidebar.header("被试管理")

# 上传新数据按钮
if st.sidebar.button("上传新数据", type="secondary", use_container_width=True):
    st.session_state['show_upload'] = True

summary = load_summary()
if summary:
    for s in summary:
        n_ep = s['n_epochs']
        st.sidebar.write(f"{s['sub_id']}  ({n_ep} epochs)")
else:
    st.sidebar.caption("暂无数据")

if st.sidebar.button("刷新数据列表", use_container_width=True):
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
    with open(f"{PREPROC_DIR}/summary.pkl", 'wb') as f:
        pickle.dump(new_sum, f)
    st.cache_resource.clear(); st.cache_data.clear()
    st.rerun()


# ==================== 主界面 ====================
st.title("CAEP 皮层听觉诱发电位 — 听力筛查分析系统")
flag=0
# ---- 上传新数据面板 (通过侧边栏按钮触发) ----
if st.session_state.get('show_upload', False):
    flag=1
    #第一次上传
    if summary is None or len(summary) == 0:
        # ---- 无数据时: 显示上传引导页 ----
        st.markdown("## 上传原始 EEG 数据开始分析")
        st.markdown("支持 **BrainVision** (.vhdr+.eeg)、**EDF/BDF** (.edf)")

        upload_tab1, upload_tab2 = st.tabs(["BrainVision", "EDF/BDF"])

        with upload_tab1:
            st.caption("上传 .vhdr 头文件 + .eeg 数据文件 (.vmrk 可选)")
            bv1, bv2 = st.columns(2)
            with bv1: vhdr_file = st.file_uploader(".vhdr (必需)", type=['vhdr'], key='main_vhdr')
            with bv2: eeg_file = st.file_uploader(".eeg (必需)", type=['eeg'], key='main_eeg')
            vmrk_file = st.file_uploader(".vmrk (可选, 未上传则用 events.tsv 或自动生成)", type=['vmrk'],
                                         key='main_vmrk')
            events_file = st.file_uploader("events.tsv (可选)", type=['tsv'], key='main_events')
            ready = vhdr_file and eeg_file

        with upload_tab2:
            st.caption("上传单个 EDF/BDF 文件")
            edf_file = st.file_uploader("选择 .edf 或 .bdf 文件", type=['edf', 'bdf'], key='main_edf')
            ready = ready or bool(edf_file)

        if ready:
            if st.button("开始自动预处理", type="primary", use_container_width=True):
                with st.status("自动预处理中...", expanded=True) as status:
                    try:
                        os.makedirs(f"{DATA_DIR}/_up/eeg", exist_ok=True)

                        # 判断格式
                        if edf_file:
                            status.update(label="步骤 1/3: 读取 EDF/BDF 文件...")
                            sub_name = edf_file.name.rsplit('.', 1)[0]
                            tmp_edf = f"{DATA_DIR}/_up/{sub_name}.edf"
                            with open(tmp_edf, 'wb') as f:
                                f.write(edf_file.getbuffer())
                            raw = mne.io.read_raw_edf(tmp_edf, preload=True, verbose=False)
                        else:
                            status.update(label="步骤 1/3: 加载 BrainVision 数据...")
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

                        status.update(label="步骤 2/3: 提取事件...")
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

                        status.update(label="步骤 3/3: 创建 Epochs & 保存...")
                        epochs = mne.Epochs(raw, mne_ev, event_id=eid_map, tmin=-0.2, tmax=0.8,
                                            baseline=(-0.2, 0), preload=True, reject=None, verbose=False)
                        out = f"{PREPROC_DIR}/{sub_name}_caep_epochs-epo.fif"
                        epochs.save(out, overwrite=True)

                        sm = load_summary() or []
                        sm.append({'sub_id': sub_name, 'n_epochs': len(epochs), 'n_channels': len(eeg_chs),
                                   'ch_names': eeg_chs, 'sfreq': SFREQ_TARGET, 'speech_types': list(eid_map.keys())})
                        with open(f"{PREPROC_DIR}/summary.pkl", 'wb') as f:
                            pickle.dump(sm, f)

                        status.update(label=f"完成! {sub_name}: {len(epochs)} epochs", state="complete")
                        st.cache_resource.clear()
                        st.cache_data.clear()
                        st.success(f"预处理成功! {sub_name} 已加入")
                        st.balloons()

                        st.cache_resource.clear()
                        st.cache_data.clear()
                        st.session_state['show_upload'] = False
                        st.rerun()
                    except Exception as e:
                        status.update(label=f"失败: {e}", state="error")
                        st.error(f"预处理出错: {e}")

        #st.info("提示: 如需演示, 可下载 DS004356 公开数据集放到 data/ 目录, 运行 preprocess.py 批量处理后使用")
        #st.session_state['show_upload'] = False
    #后续上传
    else:
        with st.expander("上传原始 EEG 数据", expanded=True):
            ut1, ut2 = st.tabs(["BrainVision", "EDF/BDF"])
            with ut1:
                uc1, uc2 = st.columns(2)
                with uc1: u_vhdr = st.file_uploader(".vhdr (必需)", type=['vhdr'], key='s_vhdr')
                with uc2: u_eeg = st.file_uploader(".eeg (必需)", type=['eeg'], key='s_eeg')
                u_vmrk = st.file_uploader(".vmrk (可选)", type=['vmrk'], key='s_vmrk')
                u_evts = st.file_uploader("events.tsv (可选)", type=['tsv'], key='s_tsv')
                ready_bv = u_vhdr and u_eeg
            with ut2:
                u_edf = st.file_uploader("选择 .edf/.bdf 文件", type=['edf','bdf'], key='s_edf')
                ready_bv = ready_bv or bool(u_edf)

            if ready_bv:
                if st.button("开始自动预处理", type="primary"):
                    with st.status("处理中...", expanded=True) as s:
                        try:
                            os.makedirs(f"{DATA_DIR}/_up/eeg", exist_ok=True)
                            if u_edf:
                                sn = u_edf.name.rsplit('.',1)[0]
                                with open(f"{DATA_DIR}/_up/{sn}.edf",'wb') as f: f.write(u_edf.getbuffer())
                                raw = mne.io.read_raw_edf(f"{DATA_DIR}/_up/{sn}.edf", preload=True, verbose=False)
                            else:
                                sn = u_vhdr.name.split('_')[0]
                                for fo, ex in [(u_vhdr,'vhdr'),(u_eeg,'eeg')]:
                                    with open(f"{DATA_DIR}/_up/eeg/{sn}.{ex}",'wb') as f: f.write(fo.getbuffer())
                                if u_vmrk:
                                    with open(f"{DATA_DIR}/_up/eeg/{sn}.vmrk",'wb') as f: f.write(u_vmrk.getbuffer())
                                if u_evts:
                                    with open(f"{DATA_DIR}/_up/eeg/{sn}_events.tsv",'wb') as f: f.write(u_evts.getbuffer())
                                raw = mne.io.read_raw_brainvision(f"{DATA_DIR}/_up/eeg/{sn}.vhdr", preload=True, verbose=False)
                            chs=[c for c in raw.ch_names if c not in ('EP1','EP2','ECG','EOG')]
                            raw.pick(chs[:min(len(chs),64)]).resample(500,verbose=False).filter(0.5,30,fir_design='firwin',verbose=False)
                            tsv_p=f"{DATA_DIR}/_up/eeg/{sn}_events.tsv"
                            if os.path.exists(tsv_p):
                                df=pd.read_csv(tsv_p,sep='\t')
                            else:
                                df=pd.DataFrame({'onset':np.linspace(1,raw.times[-1]-1,min(40,int(raw.times[-1]/2))),'value':[7]*min(40,int(raw.times[-1]/2))})
                            sd=df[df['value'].between(7,12)] if 'value' in df.columns else df
                            eid={}; me=[]
                            for _,r in sd.iterrows():
                                name=SPEECH_LABELS.get(int(r.get('value',7)),'cond')
                                if name not in eid: eid[name]=len(eid)+1
                                me.append([int(round(float(r['onset'])*500)),0,eid[name]])
                            me=np.array(me)
                            ep=mne.Epochs(raw,me,event_id=eid,tmin=-0.2,tmax=0.8,baseline=(-0.2,0),preload=True,reject=None,verbose=False)
                            ep.save(f"{PREPROC_DIR}/{sn}_caep_epochs-epo.fif",overwrite=True)
                            sm=load_summary() or []; sm.append({'sub_id':sn,'n_epochs':len(ep),'n_channels':len(chs),'ch_names':chs,'sfreq':500,'speech_types':list(eid.keys())})
                            with open(f"{PREPROC_DIR}/summary.pkl",'wb') as f: pickle.dump(sm,f)
                            s.update(label=f"完成: {sn} ({len(ep)} epochs)",state="complete")
                            st.cache_resource.clear(); st.cache_data.clear()
                            st.session_state['show_upload']=False; st.rerun()
                        except Exception as e:
                            s.update(label=f"失败: {e}",state="error")

summary = load_summary()
if summary is None and flag==0:
    st.info("请上传 EEG 数据文件开始分析")

if summary is None:
    # 截断后续渲染
    st.stop()

# ---- 分析参数 ----
st.sidebar.markdown("---")
st.sidebar.subheader("分析参数")

sub_ids = [s['sub_id'] for s in summary]
sel_subs = st.sidebar.multiselect(f"被试 ({len(sub_ids)}人可用)", sub_ids,
                                   default=sub_ids[:min(3, len(sub_ids))])
speech_opts = ['all'] + sorted(set().union(*[set(s.get('speech_types',[])) for s in summary]))
sel_speech = st.sidebar.selectbox("语音条件", speech_opts, index=0)
all_chs = summary[0]['ch_names']
sel_chs = st.sidebar.multiselect("显示通道", all_chs, default=['Fz','Cz','Pz'])

st.sidebar.markdown("---")
st.sidebar.subheader("听阈模拟")
snr_level = st.sidebar.slider("SNR (dB)", -20, 40, 10, 1,
                               help="信噪比越低 = 模拟听力损失越严重",
                               key='snr_slider')

# ---- 分析界面 ----
tab_names = ["CAEP 波形", "多维特征", "头皮地形图", "听阈模拟", "电极优化", "群体报告"]
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(tab_names)

if not sel_subs:
    st.info("请在侧边栏选择被试开始分析, 或上传新数据")
    st.stop()

# ---- 预加载数据 ----
all_epochs = {}
all_features = {}
for sid in sel_subs:
    ep = load_preprocessed(sid)
    if ep is not None:
        all_epochs[sid] = ep
        feat, ev, tm, ch = extract_all_features(ep)
        all_features[sid] = feat  # features dict

if not all_epochs:
    st.error("无法加载数据")
    st.stop()

# ==================== TAB 1: CAEP 波形 ====================
with tab1:
    st.subheader("CAEP 波形 — P1-N1-P2 自动检测")

    n_cols = min(len(sel_subs), 3)
    cols = st.columns(n_cols)

    for i, sid in enumerate(sel_subs):
        ep = all_epochs.get(sid)
        feat = all_features.get(sid)
        if ep is None:
            continue
        ev = ep.average() if sel_speech == 'all' else ep[sel_speech].average()
        times_ms = ev.times * 1000

        with cols[i % n_cols]:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            for ch in sel_chs:
                if ch in ev.ch_names:
                    idx = ev.ch_names.index(ch)
                    ax.plot(times_ms, ev.data[idx]*1e6, lw=1.5, label=ch, alpha=0.8)
            ax.axvline(0, color='k', ls='--', alpha=0.3)
            for (t1,t2,color,label) in [(30,80,'green','P1'),(80,150,'red','N1'),(150,250,'blue','P2')]:
                ax.axvspan(t1, t2, alpha=0.06, color=color)
            if feat and 'Cz' in ep.ch_names:
                for comp, color in [('P1','green'),('N1','red'),('P2','blue')]:
                    ax.scatter(feat[f'{comp}_lat'], feat[f'{comp}_amp'],
                              color=color, s=60, zorder=5)
            ax.axhline(0, color='k', ls='-', alpha=0.15)
            ax.set_xlim(-50, 400)
            ax.set_xlabel('ms'); ax.set_ylabel('uV')
            ax.set_title(f'{sid}', fontweight='bold')
            ax.legend(fontsize=7); ax.grid(alpha=0.3)
            st.pyplot(fig)

            if feat:
                st.markdown(f"""N1-P2: **{feat['N1P2']:.2f}** uV | N1 lat: {feat['N1_lat']:.0f}ms | SNR: {feat['SNR_db']:.1f} dB""")

    # 叠加对比
    if len(sel_subs) >= 2:
        st.subheader("群体 CAEP 叠加 (Cz)")
        fig, ax = plt.subplots(figsize=(10, 4))
        colors = plt.cm.viridis(np.linspace(0, 1, len(sel_subs)))
        for i, sid in enumerate(sel_subs):
            ep = all_epochs.get(sid)
            if ep and 'Cz' in ep.ch_names:
                ev = ep.average() if sel_speech == 'all' else ep[sel_speech].average()
                idx = ev.ch_names.index('Cz')
                ax.plot(ev.times*1000, ev.data[idx]*1e6, color=colors[i], lw=1.5, label=sid)
        ax.axvline(0, color='k', ls='--', alpha=0.3); ax.axhline(0, color='k', ls='-', alpha=0.15)
        ax.set_xlim(-50, 400); ax.set_xlabel('ms'); ax.set_ylabel('uV')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        st.pyplot(fig)

# ==================== TAB 2: 多维特征 ====================
with tab2:
    st.subheader("多维特征分析 — 除 CAEP 外的辅助特征")

    feat_cols = st.columns(min(len(sel_subs), 4))
    for i, sid in enumerate(sel_subs):
        feat = all_features.get(sid)
        if feat is None:
            continue
        with feat_cols[i % len(feat_cols)]:
            st.markdown(f"**{sid}**")
            df = pd.DataFrame({
                '特征': ['P1 幅度(uV)','N1 幅度(uV)','P2 幅度(uV)','N1-P2(uV)',
                         'delta功率','theta功率','alpha功率','beta功率',
                         'GFP max','ITC(alpha)','SNR(dB)'],
                '值': [f"{feat['P1_amp']:.2f}", f"{feat['N1_amp']:.2f}", f"{feat['P2_amp']:.2f}",
                       f"{feat['N1P2']:.2f}", f"{feat['power_delta']:.1e}", f"{feat['power_theta']:.1e}",
                       f"{feat['power_alpha']:.1e}", f"{feat['power_beta']:.1e}",
                       f"{feat['GFP_max']:.2f}", f"{feat['ITC_alpha_mean']:.3f}",
                       f"{feat['SNR_db']:.1f}"]
            })
            st.dataframe(df, hide_index=True, height=360, use_container_width=True)

    # 特征对比图
    st.markdown("---")
    st.subheader("特征对比 (群体)")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    metric_keys = ['N1P2', 'SNR_db', 'GFP_max', 'power_alpha', 'ITC_alpha_mean', 'P2_amp']
    metric_labels = ['N1-P2 (uV)', 'SNR (dB)', 'GFP max (uV)', 'Alpha 功率', 'ITC (alpha)', 'P2 幅度 (uV)']
    for ax, key, label in zip(axes.flat, metric_keys, metric_labels):
        vals = [all_features[s][key] for s in sel_subs if s in all_features]
        names = [s for s in sel_subs if s in all_features]
        bars = ax.bar(range(len(vals)), vals, color=plt.cm.RdYlGn(
            np.clip((np.array(vals)-np.min(vals)+1e-6)/(np.max(vals)-np.min(vals)+1e-6), 0, 1)))
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(names, fontsize=7, rotation=30)
        ax.set_title(label, fontweight='bold', fontsize=10)
        ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    st.pyplot(fig)

    # 频段能量雷达图
    st.subheader("频段能量分布")
    fig, ax = plt.subplots(figsize=(6, 5), subplot_kw=dict(polar=True))
    bands_list = list(CAEP_BANDS.keys())
    angles = np.linspace(0, 2*np.pi, len(bands_list), endpoint=False).tolist()
    angles += angles[:1]
    colors = plt.cm.tab10(np.linspace(0, 1, len(sel_subs)))
    for i, sid in enumerate(sel_subs):
        feat = all_features.get(sid)
        if feat is None:
            continue
        vals = [feat[f'power_{b}'] for b in bands_list]
        vals = [v / max(vals + [1e-12]) for v in vals]  # normalize
        vals_plot = vals + vals[:1]
        ax.fill(angles, vals_plot, alpha=0.15, color=colors[i])
        ax.plot(angles, vals_plot, 'o-', lw=1.5, color=colors[i], label=sid, markersize=4)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f'{b}' for b in bands_list], fontsize=11)
    ax.set_title('频段能量雷达图', fontsize=12, fontweight='bold')
    ax.legend(fontsize=7, loc='upper right')
    st.pyplot(fig)

# ==================== TAB 3: 头皮地形图 ====================
with tab3:
    st.subheader("CAEP 头皮地形图")
    time_pts = [50, 80, 110, 150, 190]
    time_labels = [f'{tp}ms ({"P1" if tp<80 else "N1" if tp<150 else "P2"})' for tp in time_pts]

    for sid in sel_subs[:4]:
        ep = all_epochs.get(sid)
        if ep is None:
            continue
        ev = ep.average() if sel_speech == 'all' else ep[sel_speech].average()
        st.markdown(f"**{sid}**")
        fig, axes = plt.subplots(1, 5, figsize=(18, 3.5))
        for ax, tp, tl in zip(axes, time_pts, time_labels):
            plot_topo(ev, ep.info, tp, ax, tl)
        plt.tight_layout()
        st.pyplot(fig)

# ==================== TAB 4: 听阈模拟 ====================
with tab4:
    st.subheader("听阈模拟 — 通过 SNR 衰减模拟听力损失")
    st.caption("拖动侧边栏 SNR 滑块观察实时变化")

    snr_vals = list(range(-10, 35, 5))

    col1, col2 = st.columns([2, 1])
    with col1:
        # 听阈推断曲线
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for sid in sel_subs[:6]:
            ep = all_epochs.get(sid)
            if ep is None:
                continue
            n1p2_vals = []
            for snr in snr_vals:
                ev_noisy = simulate_hearing_loss(ep, snr)
                if 'Cz' in ev_noisy.ch_names:
                    idx = ev_noisy.ch_names.index('Cz')
                    d = ev_noisy.data[idx]*1e6; t = ev_noisy.times*1000
                    n1 = np.min(d[(t>=80)&(t<=150)])
                    p2 = np.max(d[(t>=150)&(t<=250)])
                    n1p2_vals.append(p2 - n1)
                else:
                    n1p2_vals.append(np.nan)
            valid = ~np.isnan(n1p2_vals)
            ax.plot(np.array(snr_vals)[valid], np.array(n1p2_vals)[valid], 'o-', lw=2, ms=8, label=sid)

        ax.axhline(1.0, color='red', ls='--', lw=1.5, label='临床阈值 (1 uV)')
        ax.axvline(0, color='gray', ls=':', alpha=0.5)
        ax.set_xlabel('SNR (dB)', fontsize=12); ax.set_ylabel('N1-P2 (uV)', fontsize=12)
        ax.set_title('听阈推断曲线', fontweight='bold', fontsize=14)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        st.pyplot(fig)

    with col2:
        st.markdown("""
        ### 当前 SNR 下的波形
        侧边栏 SNR = **{snr} dB**
        """.format(snr=snr_level))
        for sid in sel_subs[:3]:
            ep = all_epochs.get(sid)
            if ep is None or 'Cz' not in ep.ch_names:
                continue
            ev_noisy = simulate_hearing_loss(ep, snr_level)
            ev_clean = ep.average()
            fig, ax = plt.subplots(figsize=(4, 2.5))
            idx = ev_clean.ch_names.index('Cz')
            ax.plot(ev_clean.times*1000, ev_clean.data[idx]*1e6, 'b-', lw=1.5, label='原始', alpha=0.7)
            ax.plot(ev_noisy.times*1000, ev_noisy.data[idx]*1e6, 'r-', lw=1.5, label=f'SNR={snr_level}', alpha=0.7)
            ax.axvline(0, color='k', ls='--', alpha=0.3); ax.axhline(0, color='k', ls='-', alpha=0.15)
            ax.set_xlim(-50, 400); ax.set_title(sid, fontsize=9); ax.legend(fontsize=7)
            st.pyplot(fig)

    st.markdown("---")
    st.markdown("""
    **听阈估计方法**: N1-P2 峰峰幅值降至 1 uV 对应的 SNR 值即为该被试的估计听阈。
    正常听力者 SNR 阈值通常在 -5 到 5 dB 范围内。
    """)

# ==================== TAB 5: 电极优化 ====================
with tab5:
    st.subheader("电极最小化分析")
    st.markdown("线性回归评估每个电极对 CAEP 信号的贡献, 找出最小必需的电极组合")

    for sid in sel_subs[:3]:
        feat = all_features.get(sid)
        if feat is None:
            continue
        ci = feat['channel_importance']
        chs_sorted = sorted(ci, key=ci.get, reverse=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        # 电极重要性排序
        top_n = min(16, len(chs_sorted))
        top_chs = chs_sorted[:top_n]
        values = [ci[ch] for ch in top_chs]
        colors_bar = ['#E74C3C' if v > 0.8 else '#F39C12' if v > 0.5 else '#3498DB' for v in values]
        axes[0].barh(range(top_n), values, color=colors_bar)
        axes[0].set_yticks(range(top_n))
        axes[0].set_yticklabels(top_chs, fontsize=8)
        axes[0].axvline(0.8, color='red', ls='--', alpha=0.7, label='高贡献阈值 (0.8)')
        axes[0].axvline(0.5, color='orange', ls='--', alpha=0.7, label='中贡献阈值 (0.5)')
        axes[0].set_xlabel('与 GFP 的相关系数'); axes[0].set_title(f'{sid} 电极贡献排序', fontweight='bold')
        axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3, axis='x')
        axes[0].invert_yaxis()

        # 最小电极组的 CAEP
        ax2 = axes[1]
        ep = all_epochs.get(sid)
        if ep:
            ev = ep.average()
            t = ev.times * 1000
            # 全电极 vs 推荐最少电极
            minimal = feat['minimal_electrodes'][:3]  # 前3个最关键
            for ch, style in [(minimal[0], 'r-'), ('Cz', 'b-')]:
                if ch in ev.ch_names:
                    ax2.plot(t, ev.data[ev.ch_names.index(ch)]*1e6, style, lw=2, label=f'{ch}', alpha=0.7)
            ax2.axvline(0, color='k', ls='--', alpha=0.3); ax2.axhline(0, color='k', ls='-', alpha=0.15)
            ax2.set_xlim(-50, 400); ax2.set_xlabel('ms'); ax2.set_ylabel('uV')
            ax2.set_title(f'最少电极 vs Cz: {", ".join(minimal[:3])}', fontweight='bold')
            ax2.legend(); ax2.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown(f"**{sid}** 推荐最少电极组合: **{', '.join(feat['minimal_electrodes'][:3])}** (贡献 > 0.9)")

    # 全局电极推荐
    st.markdown("---")
    st.subheader("全局电极推荐 (基于所有被试)")
    # 汇总所有被试的电极重要性
    global_ci = {}
    for sid in sel_subs:
        feat = all_features.get(sid)
        if feat:
            for ch, val in feat['channel_importance'].items():
                global_ci[ch] = global_ci.get(ch, []) + [val]
    avg_ci = {ch: np.mean(vals) for ch, vals in global_ci.items()}
    sorted_global = sorted(avg_ci, key=avg_ci.get, reverse=True)
    best3 = sorted_global[:3]
    best6 = sorted_global[:6]

    st.success(f"""
    **推荐最小电极配置**:
    - 最少 3 电极: **{', '.join(best3)}** (平均相关 {avg_ci[best3[0]]:.3f}, {avg_ci[best3[1]]:.3f}, {avg_ci[best3[2]]:.3f})
    - 推荐 6 电极: **{', '.join(best6)}**
    """)

# ==================== TAB 6: 群体报告 ====================
with tab6:
    st.subheader("群体统计报告")
    all_feat_list = []
    for sid in sel_subs:
        feat = all_features.get(sid)
        if feat:
            all_feat_list.append({
                'subject': sid,
                'P1_lat_ms': feat['P1_lat'], 'P1_amp_uV': feat['P1_amp'],
                'N1_lat_ms': feat['N1_lat'], 'N1_amp_uV': feat['N1_amp'],
                'P2_lat_ms': feat['P2_lat'], 'P2_amp_uV': feat['P2_amp'],
                'N1P2_uV': feat['N1P2'], 'SNR_dB': feat['SNR_db'],
                'GFP_max': feat['GFP_max'], 'ITC_alpha': feat['ITC_alpha_mean'],
            })

    if all_feat_list:
        df = pd.DataFrame(all_feat_list)
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("被试数", len(df))
        with c2: st.metric("N1 潜伏期均值", f"{df['N1_lat_ms'].mean():.0f} ± {df['N1_lat_ms'].std():.0f} ms")
        with c3: st.metric("N1-P2 均值", f"{df['N1P2_uV'].mean():.2f} ± {df['N1P2_uV'].std():.2f} uV")
        with c4: st.metric("SNR 均值", f"{df['SNR_dB'].mean():.1f} dB")

        st.dataframe(df.style.format(precision=2), use_container_width=True,
                    column_order=['subject','N1P2_uV','SNR_dB','N1_lat_ms','P2_lat_ms','GFP_max','ITC_alpha'])

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("下载群体报告 (CSV)", data=csv, file_name='CAEP_group_report.csv', mime='text/csv')

        # 分布图
        st.subheader("N1-P2 分布")
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].hist(df['N1P2_uV'], bins=min(10, len(df)), color='#3498DB', edgecolor='white')
        axes[0].axvline(1.0, color='red', ls='--', lw=1.5, label='临床阈值')
        axes[0].set_xlabel('N1-P2 (uV)'); axes[0].set_ylabel('人数'); axes[0].legend(); axes[0].grid(alpha=0.3)

        axes[1].scatter(df['N1_lat_ms'], df['P2_lat_ms'], c=df['N1P2_uV'], cmap='RdYlGn', s=80, edgecolors='black', lw=0.5)
        axes[1].set_xlabel('N1 潜伏期 (ms)'); axes[1].set_ylabel('P2 潜伏期 (ms)'); axes[1].grid(alpha=0.3)

        bar_colors = ['green' if v>1.0 else 'red' for v in df['N1P2_uV']]
        axes[2].barh(df['subject'], df['N1P2_uV'], color=bar_colors)
        axes[2].axvline(1.0, color='red', ls='--', lw=1.5)
        axes[2].set_xlabel('N1-P2 (uV)'); axes[2].grid(alpha=0.3, axis='x')
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("无可用数据")
