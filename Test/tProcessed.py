import mne
import numpy as np
import pandas as pd
import pickle
import os
from scipy import signal

SFREQ = 500
PREPROC_DIR = "preprocessed"
os.makedirs(PREPROC_DIR, exist_ok=True)

def process_bdf(file_path):
    # 1. 读取 BDF
    raw = mne.io.read_raw_bdf(file_path, preload=True, verbose=False)

    # 2. 选择 EEG 通道
    eeg_chs = [ch for ch in raw.ch_names
               if raw.get_channel_types(picks=[ch])[0] == 'eeg'
               and ch not in ('EP1','EP2','ECG','EOG')]
    if not eeg_chs:
        # 备用：取所有非明确非 EEG 通道
        eeg_chs = [ch for ch in raw.ch_names if ch not in ('ECG','EOG','STI 014')]
    raw.pick(eeg_chs)
    raw.resample(SFREQ, verbose=False)
    raw.filter(0.5, 30, fir_design='firwin', verbose=False)

    # 3. 提取事件（更健壮）
    try:
        # 尝试从 annotations 提取
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        # 如果 event_id 中有 7~12，则筛选；否则保留全部
        speech_values = [v for v in event_id.values() if 7 <= v <= 12]
        if speech_values:
            mask = np.isin(events[:, 2], speech_values)
            speech_events = events[mask]
        else:
            # 没有语音事件，则使用全部事件（或生成虚拟事件）
            print("未找到语音事件 (7-12)，使用全部事件")
            speech_events = events
        if len(speech_events) == 0:
            # 如果仍然为空，生成虚拟事件
            print("没有可用事件，生成虚拟事件")
            n = int(raw.times[-1] // 2)
            onset_times = np.linspace(1, raw.times[-1]-1, n)
            speech_events = np.array([[int(round(t*SFREQ)), 0, 7] for t in onset_times])
            event_id = {'speech': 7}
    except:
        # 若 annotations 提取失败，生成虚拟事件
        print("无法从 annotations 提取事件，生成虚拟事件")
        n = int(raw.times[-1] // 2)
        onset_times = np.linspace(1, raw.times[-1]-1, n)
        speech_events = np.array([[int(round(t*SFREQ)), 0, 7] for t in onset_times])
        event_id = {'speech': 7}

    # 4. 创建 Epochs（使用实际事件 ID）
    epochs = mne.Epochs(raw, speech_events, event_id=event_id,
                        tmin=-0.2, tmax=0.8, baseline=(-0.2,0),
                        preload=True, reject=None, verbose=False)

    # 5. 保存
    sub_name = os.path.splitext(os.path.basename(file_path))[0]
    out_path = f"{PREPROC_DIR}/{sub_name}_caep_epochs-epo.fif"
    epochs.save(out_path, overwrite=True)

    # 6. 更新 summary
    summary_path = f"{PREPROC_DIR}/summary.pkl"
    if os.path.exists(summary_path):
        with open(summary_path, 'rb') as f:
            summary = pickle.load(f)
    else:
        summary = []
    summary.append({
        'sub_id': sub_name,
        'n_epochs': len(epochs),
        'n_channels': len(eeg_chs),
        'ch_names': eeg_chs,
        'sfreq': SFREQ,
        'speech_types': list(event_id.keys())
    })
    with open(summary_path, 'wb') as f:
        pickle.dump(summary, f)

    print(f"✅ 处理完成：{sub_name}，Epochs 数：{len(epochs)}")

if __name__ == "__main__":
    # 请将你的 BDF 文件路径放在这里
    bdf_file = "data/1.bdf"   # 修改为你的文件路径
    process_bdf(bdf_file)