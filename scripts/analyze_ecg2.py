
import sys, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch, find_peaks, welch

sys.stdout.reconfigure(encoding='utf-8')

with open(r'c:\Users\kanhaiya panjiyar\.gemini\antigravity\scratch\my-workspace\CardiogastroVest\Device_0_Volts.xls', 'r', errors='replace') as f:
    content = f.read()

volts_start = content.find('Volts Data')
data_section = content[volts_start:]
values = re.findall(r'[-+]?\d+\.\d+E[+-]\d+', data_section)
raw = np.array([float(v) for v in values])
fs = 250
t = np.arange(len(raw)) / fs

# Apply filters
b, a = butter(3, 0.5/(fs/2), btype='high')
sig = filtfilt(b, a, raw)
b2, a2 = iirnotch(50, 30, fs)
sig = filtfilt(b2, a2, sig)
b3, a3 = butter(3, 40/(fs/2), btype='low')
sig = filtfilt(b3, a3, sig)

# Focus on CLEAN SECTION: 3s to 16s
start, end = int(3*fs), int(16*fs)
t_clean = t[start:end]
sig_clean = sig[start:end]

print("=== HONEST ECG ANALYSIS ===")
print(f"Clean section: 3s to 16s ({len(sig_clean)/fs:.1f} seconds)")
print(f"Amplitude: {sig_clean.min()*1000:.2f} mV to {sig_clean.max()*1000:.2f} mV")
print(f"RMS noise: {sig_clean.std()*1000:.2f} mV")
print(f"Expected ECG amplitude at gain=1: 0.5 to 2.0 mV")

# Power spectral density of clean section
freqs, psd = welch(sig_clean, fs, nperseg=512)
print(f"\nPSD Analysis (clean section):")
# Find peak in heartbeat frequency range 0.5-3 Hz
hr_mask = (freqs >= 0.5) & (freqs <= 3.0)
if hr_mask.any():
    hr_peak_idx = np.argmax(psd[hr_mask])
    hr_freqs = freqs[hr_mask]
    peak_freq = hr_freqs[hr_peak_idx]
    print(f"Dominant frequency in HR range (0.5-3 Hz): {peak_freq:.2f} Hz = {peak_freq*60:.0f} BPM")
    print(f"PSD at peak: {10*np.log10(psd[hr_mask][hr_peak_idx]):.1f} dBV²/Hz")

# Peak in EMG range 5-40 Hz
emg_mask = (freqs >= 5) & (freqs <= 40)
emg_power = np.mean(psd[emg_mask])
ecg_power = np.mean(psd[hr_mask])
print(f"\nPower comparison:")
print(f"HR band (0.5-3 Hz) mean PSD: {10*np.log10(ecg_power):.1f} dB")
print(f"EMG band (5-40 Hz) mean PSD: {10*np.log10(emg_power):.1f} dB")
snr = 10*np.log10(ecg_power/emg_power)
print(f"ECG SNR estimate: {snr:.1f} dB")

# Verdict
print("\n=== VERDICT ===")
if sig_clean.std()*1000 < 3:
    print("Signal amplitude: ~noise level")
if snr < 0:
    print("EMG power > ECG band power → WRIST MUSCLE NOISE DOMINANT")
    print("Clear ECG: NOT confirmed from this recording")
    print("Solution needed: Chest electrodes + higher gain (with proper setup)")
else:
    print("ECG band is stronger than EMG → possible ECG present")

# PLOT
fig, axes = plt.subplots(3, 1, figsize=(14, 10))
fig.suptitle('Honest ECG Analysis — Device_0_Volts.xls', fontsize=13, fontweight='bold', color='darkred')

# Full filtered signal
axes[0].plot(t, sig*1000, 'steelblue', linewidth=0.5)
axes[0].axvspan(3, 16, alpha=0.1, color='green', label='Clean analysis window')
axes[0].axvspan(16, t[-1], alpha=0.1, color='red', label='Artifact (electrode moved)')
axes[0].set_title('Full Filtered Signal')
axes[0].set_ylabel('mV')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# Clean section zoomed
axes[1].plot(t_clean, sig_clean*1000, 'darkgreen', linewidth=0.8)
axes[1].set_title('Clean Section (3-16s) — Looking for ECG (should show ~1 Hz periodic spikes)')
axes[1].set_ylabel('mV')
axes[1].set_xlabel('Time (s)')
axes[1].grid(True, alpha=0.3)
axes[1].axhline(0, color='gray', linewidth=0.5)

# PSD
axes[2].semilogy(freqs, psd)
axes[2].axvspan(0.5, 3, alpha=0.2, color='green', label='ECG band (0.5-3 Hz)')
axes[2].axvspan(5, 40, alpha=0.1, color='red', label='EMG/noise band')
axes[2].axvline(50, color='orange', linestyle='--', linewidth=0.8, label='50 Hz (notched)')
axes[2].set_title('Power Spectral Density — CH1')
axes[2].set_xlabel('Frequency (Hz)')
axes[2].set_ylabel('PSD (V²/Hz)')
axes[2].set_xlim(0, 125)
axes[2].legend(fontsize=8)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(r'c:\Users\kanhaiya panjiyar\.gemini\antigravity\scratch\my-workspace\CardiogastroVest\ecg_honest_analysis.png', dpi=150, bbox_inches='tight')
print("\nPlot saved.")
