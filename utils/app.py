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
import warnings, traceback
import matplotlib.pyplot as plt
import matplotlib

import sys
from pathlib import Path



# 将项目根目录（CAEP_analysis）添加到 Python 路径
sys.path.append(str(Path(__file__).parent.parent))

# 现在可以正常使用绝对导入
from utils.plotting import *
from utils.processing import *
from utils.load import *

# 设置matplotlib正常显示中文和负号
matplotlib.rcParams['font.family'] = 'SimHei'  # 'SimHei' 或其他支持中文的字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 为了防止乱码，也可以指定使用多个字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号
warnings.filterwarnings('ignore')

st.set_page_config(page_title="CAEP 听力筛查分析系统", layout="wide", initial_sidebar_state="expanded")

DATA_DIR = "data"
PREPROC_DIR = "preprocessed"
os.makedirs(PREPROC_DIR, exist_ok=True)
SFREQ_TARGET = 500
SPEECH_LABELS = {7:'chn_aud',8:'eng_aud',9:'interview',10:'lecture',11:'news',12:'talk'}
CAEP_BANDS = {'delta':(1,4), 'theta':(4,8), 'alpha':(8,13), 'beta':(13,30)}

# ==================== 侧边栏 ====================
st.sidebar.header("被试管理")

if 'flag' not in st.session_state:
    st.session_state.flag = 0
if 'flag1' not in st.session_state:
    st.session_state.flag1 = 0

# 上传新数据按钮
if st.sidebar.button("上传新数据", type="secondary", use_container_width=True):
    st.session_state['show_upload'] = True

summary = load_summary(PREPROC_DIR)
if summary:
    for s in summary:
        n_ep = s['n_epochs']
        st.sidebar.write(f"{s['sub_id']}  ({n_ep} epochs)")
else:
    st.sidebar.caption("暂无数据")

if st.sidebar.button("刷新数据列表", use_container_width=True):
    new_sum=fif_Load(PREPROC_DIR)
    with open(f"{PREPROC_DIR}/summary.pkl", 'wb') as f:
        pickle.dump(new_sum, f)
    if new_sum:
        st.session_state.flag1=1
    st.cache_resource.clear()
    st.cache_data.clear()
    st.rerun()


# ==================== 主界面 ====================
st.title("CAEP 皮层听觉诱发电位 — 听力筛查分析系统")

# ---- 上传新数据面板 (通过侧边栏按钮触发) ----
if st.session_state.get('show_upload', False):
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

    if ready and st.button("开始自动预处理", type="primary", use_container_width=True):
        with st.status("自动预处理中...", expanded=True) as status:
            sub_name, epochs = auto_preprocessing(DATA_DIR, SFREQ_TARGET, SPEECH_LABELS, PREPROC_DIR,
                                                  edf_file, vhdr_file, eeg_file, vmrk_file, events_file)
            status.update(label=f"完成! {sub_name}: {len(epochs)} epochs", state="complete")
            st.cache_resource.clear()
            st.cache_data.clear()
            st.success(f"预处理成功! {sub_name} 已加入")
            st.balloons()

            st.cache_resource.clear()
            st.cache_data.clear()
            st.session_state.flag = 1
            st.session_state['show_upload'] = False
            st.rerun()
        #st.info("提示: 如需演示, 可下载 DS004356 公开数据集放到 data/ 目录, 运行 preprocess.py 批量处理后使用")
        #st.session_state['show_upload'] = False

# ---- 显示侧边栏 ----
if st.session_state.flag or st.session_state.flag1 :#第一次上传
    st.sidebar.markdown("---")
    st.sidebar.subheader("分析参数")
    from utils.load import load_summary

    summary = load_summary(PREPROC_DIR)

    sub_ids = [s['sub_id'] for s in summary]
    sel_subs = st.sidebar.multiselect(f"被试 ({len(sub_ids)}人可用)", sub_ids,
                                      default=sub_ids[:min(3, len(sub_ids))])
    speech_opts = ['all'] + sorted(set().union(*[set(s.get('speech_types', [])) for s in summary]))
    sel_speech = st.sidebar.selectbox("语音条件", speech_opts, index=0)
    all_chs = summary[0]['ch_names']
    sel_chs = st.sidebar.multiselect("显示通道", all_chs, default=['Fz', 'Cz', 'Pz'])

    st.sidebar.markdown("---")
    st.sidebar.subheader("听阈模拟")
    snr_level = st.sidebar.slider("SNR (dB)", -20, 40, 10, 1,
                                  help="信噪比越低 = 模拟听力损失越严重",
                                  key='snr_slider')
#不显示
else:
    # 截断后续渲染
    st.info("请上传 EEG 数据文件开始分析")
    st.stop()

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
    ep = load_preprocessed(sid,PREPROC_DIR)
    if ep is not None:
        all_epochs[sid] = ep
        feat, ev, tm, ch = extract_all_features(ep,CAEP_BANDS)
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
            fig=CAEP_Ploting(sel_chs,times_ms,ev,ep,feat,sid)
            st.pyplot(fig)
            if feat:st.markdown(f"""N1-P2: **{feat['N1P2']:.2f}** uV | N1 lat: {feat['N1_lat']:.0f}ms | SNR: {feat['SNR_db']:.1f} dB""")

    # 叠加对比
    if len(sel_subs) >= 2:
        st.subheader("群体 CAEP 叠加 (Cz)")
        fig=CAEP_STACK(sel_subs,all_epochs,sel_speech)
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
    fig=Feature_Compare(all_features,sel_subs)
    st.pyplot(fig)

    # 频段能量雷达图
    st.subheader("频段能量分布")
    fig=Radio_Plot(CAEP_BANDS,sel_subs,all_features)
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
        fig=Audio_Threshold(sel_subs,all_epochs,snr_vals)
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
        fig=Ele_opt(feat,sid,all_epochs)
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
        fig=N1P2(df)
        st.pyplot(fig)
    else:
        st.warning("无可用数据")
