import numpy as np
from matplotlib import pyplot as plt



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


def CAEP_Ploting(sel_chs,times_ms,ev,ep,feat,sid):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for ch in sel_chs:
        if ch in ev.ch_names:
            idx = ev.ch_names.index(ch)
            ax.plot(times_ms, ev.data[idx] * 1e6, lw=1.5, label=ch, alpha=0.8)
    ax.axvline(0, color='k', ls='--', alpha=0.3)
    for (t1, t2, color, label) in [(30, 80, 'green', 'P1'), (80, 150, 'red', 'N1'), (150, 250, 'blue', 'P2')]:
        ax.axvspan(t1, t2, alpha=0.06, color=color)
    if feat and 'Cz' in ep.ch_names:
        for comp, color in [('P1', 'green'), ('N1', 'red'), ('P2', 'blue')]:
            ax.scatter(feat[f'{comp}_lat'], feat[f'{comp}_amp'],
                       color=color, s=60, zorder=5)
    ax.axhline(0, color='k', ls='-', alpha=0.15)
    ax.set_xlim(-50, 400)
    ax.set_xlabel('ms')
    ax.set_ylabel('uV')
    ax.set_title(f'{sid}', fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    return fig

def CAEP_STACK(sel_subs,all_epochs,sel_speech):
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = plt.cm.viridis(np.linspace(0, 1, len(sel_subs)))
    for i, sid in enumerate(sel_subs):
        ep = all_epochs.get(sid)
        if ep and 'Cz' in ep.ch_names:
            ev = ep.average() if sel_speech == 'all' else ep[sel_speech].average()
            idx = ev.ch_names.index('Cz')
            ax.plot(ev.times * 1000, ev.data[idx] * 1e6, color=colors[i], lw=1.5, label=sid)
    ax.axvline(0, color='k', ls='--', alpha=0.3)
    ax.axhline(0, color='k', ls='-', alpha=0.15)
    ax.set_xlim(-50, 400)
    ax.set_xlabel('ms')
    ax.set_ylabel('uV')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return fig

def Feature_Compare(all_features,sel_subs):
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

    return fig

def Radio_Plot(CAEP_BANDS,sel_subs,all_features):
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
    return fig

def Audio_Threshold(sel_subs,all_epochs,snr_vals):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for sid in sel_subs[:6]:
        ep = all_epochs.get(sid)
        if ep is None:
            continue
        n1p2_vals = []
        for snr in snr_vals:
            from utils.processing import simulate_hearing_loss
            ev_noisy = simulate_hearing_loss(ep, snr)
            if 'Cz' in ev_noisy.ch_names:
                idx = ev_noisy.ch_names.index('Cz')
                d = ev_noisy.data[idx] * 1e6
                t = ev_noisy.times * 1000
                n1 = np.min(d[(t >= 80) & (t <= 150)])
                p2 = np.max(d[(t >= 150) & (t <= 250)])
                n1p2_vals.append(p2 - n1)
            else:
                n1p2_vals.append(np.nan)
        valid = ~np.isnan(n1p2_vals)
        ax.plot(np.array(snr_vals)[valid], np.array(n1p2_vals)[valid], 'o-', lw=2, ms=8, label=sid)

    ax.axhline(1.0, color='red', ls='--', lw=1.5, label='临床阈值 (1 uV)')
    ax.axvline(0, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('N1-P2 (uV)', fontsize=12)
    ax.set_title('听阈推断曲线', fontweight='bold', fontsize=14)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return fig

def Ele_opt(feat,sid,all_epochs):
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
    axes[0].set_xlabel('与 GFP 的相关系数')
    axes[0].set_title(f'{sid} 电极贡献排序', fontweight='bold')
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3, axis='x')
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
                ax2.plot(t, ev.data[ev.ch_names.index(ch)] * 1e6, style, lw=2, label=f'{ch}', alpha=0.7)
        ax2.axvline(0, color='k', ls='--', alpha=0.3)
        ax2.axhline(0, color='k', ls='-', alpha=0.15)
        ax2.set_xlim(-50, 400)
        ax2.set_xlabel('ms')
        ax2.set_ylabel('uV')
        ax2.set_title(f'最少电极 vs Cz: {", ".join(minimal[:3])}', fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
    plt.tight_layout()
    return fig

def N1P2(df):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].hist(df['N1P2_uV'], bins=min(10, len(df)), color='#3498DB', edgecolor='white')
    axes[0].axvline(1.0, color='red', ls='--', lw=1.5, label='临床阈值')
    axes[0].set_xlabel('N1-P2 (uV)')
    axes[0].set_ylabel('人数')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].scatter(df['N1_lat_ms'], df['P2_lat_ms'], c=df['N1P2_uV'], cmap='RdYlGn', s=80, edgecolors='black', lw=0.5)
    axes[1].set_xlabel('N1 潜伏期 (ms)')
    axes[1].set_ylabel('P2 潜伏期 (ms)')
    axes[1].grid(alpha=0.3)

    bar_colors = ['green' if v > 1.0 else 'red' for v in df['N1P2_uV']]
    axes[2].barh(df['subject'], df['N1P2_uV'], color=bar_colors)
    axes[2].axvline(1.0, color='red', ls='--', lw=1.5)
    axes[2].set_xlabel('N1-P2 (uV)')
    axes[2].grid(alpha=0.3, axis='x')
    plt.tight_layout()
    return fig