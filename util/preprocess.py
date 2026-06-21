"""
DS004356 批量预处理 — 支持多 run 被试, 高效批处理
"""
import mne, numpy as np, os, glob, pickle, pandas as pd
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "data"
OUT_DIR = "preprocessed"
os.makedirs(OUT_DIR, exist_ok=True)
SPEECH_VALUES = {7:'chn_aud',8:'eng_aud',9:'interview',10:'lecture',11:'news',12:'talk'}
TARGET_SFREQ = 500

def preprocess_subject(sub_id):
    sub_dir = f"{DATA_DIR}/{sub_id}/eeg"
    if not os.path.isdir(sub_dir):
        return None

    # 查找所有 run 的 vhdr
    vhdr_files = sorted(glob.glob(f"{sub_dir}/*_eeg.vhdr"))
    if not vhdr_files:
        return None

    all_epochs = []
    eeg_chs = None

    for vhdr in vhdr_files:
        try:
            raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose=False)
            chs = [ch for ch in raw.ch_names
                   if raw.get_channel_types(picks=[ch])[0]=='eeg'
                   and ch not in ('EP1','EP2')]
            if eeg_chs is None:
                eeg_chs = chs
            raw.pick(chs)
            raw.resample(TARGET_SFREQ, verbose=False)
            raw.filter(0.5, 30, fir_design='firwin', verbose=False)

            # 读 events.tsv (同目录)
            events_tsv = vhdr.replace('_eeg.vhdr', '_events.tsv')
            if not os.path.exists(events_tsv):
                continue
            df = pd.read_csv(events_tsv, sep='\t')
            speech_df = df[df['value'].isin(SPEECH_VALUES.keys())]
            if len(speech_df) == 0:
                continue

            sfreq = raw.info['sfreq']
            eid = {}
            ev = []
            for _, row in speech_df.iterrows():
                onset = float(row['onset'])
                sample = int(round(onset * sfreq))
                name = SPEECH_VALUES.get(int(row['value']), f'cond_{int(row["value"])}')
                if name not in eid:
                    eid[name] = len(eid) + 1
                ev.append([sample, 0, eid[name]])
            ev = np.array(ev)

            epochs = mne.Epochs(raw, ev, event_id=eid, tmin=-0.2, tmax=0.8,
                                baseline=(-0.2, 0), preload=True, reject=None, verbose=False)
            if len(epochs) > 0:
                all_epochs.append(epochs)
        except Exception as e:
            print(f"  Warning ({vhdr}): {e}")

    if not all_epochs or eeg_chs is None:
        return None

    # 合并所有 run
    if len(all_epochs) > 1:
        combined = mne.concatenate_epochs(all_epochs, verbose=False)
    else:
        combined = all_epochs[0]

    out_path = f"{OUT_DIR}/{sub_id}_caep_epochs-epo.fif"
    combined.save(out_path, overwrite=True)

    return {
        'sub_id': sub_id, 'n_epochs': len(combined),
        'n_channels': len(eeg_chs), 'ch_names': eeg_chs,
        'sfreq': TARGET_SFREQ, 'speech_types': list(combined.event_id.keys()),
    }


def main():
    sub_dirs = sorted(glob.glob(f"{DATA_DIR}/sub-*"))
    sub_ids = [os.path.basename(d) for d in sub_dirs if os.path.isdir(d)]
    print(f"找到 {len(sub_ids)} 个被试\n")

    summary = []
    for sid in sub_ids:
        print(f"[{sid}]", end=' ')
        info = preprocess_subject(sid)
        if info:
            summary.append(info)
            print(f"OK: {info['n_epochs']} epochs, {info['n_channels']} channels")
        else:
            print("SKIP")

    with open(f"{OUT_DIR}/summary.pkl", 'wb') as f:
        pickle.dump(summary, f)
    print(f"\nDone! {len(summary)}/{len(sub_ids)} subjects. {sum(s['n_epochs'] for s in summary)} total epochs.")


if __name__ == '__main__':
    main()
