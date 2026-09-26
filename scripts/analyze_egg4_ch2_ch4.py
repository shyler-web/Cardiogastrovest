"""
CardiogastroVest — EGG DATA 4: Deep CH2 & CH4 Analysis
========================================================
Dedicated, high-accuracy per-channel analysis:
  1. Skip first 2 min
  2. sosfiltfilt 0.016–0.25 Hz (stable)
  3. Multi-resolution PSD: 2-min, 3-min, 4-min epoch averages
  4. Time-frequency spectrogram (STFT)
  5. Wave-by-wave analysis: inter-wave intervals, amplitude stats
  6. Cross-channel coherence / correlation
  7. Per-epoch dominant frequency stability (DFIC)

Author: Antigravity AI  |  2026-09-24
"""

import sys, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
from scipy.signal import (butter, sosfiltfilt, welch, find_peaks,
                           spectrogram, coherence)
import scipy.integrate

sys.stdout.reconfigure(encoding='utf-8')

DATA_FILE = r'c:\Users\kanhaiya panjiyar\.gemini\antigravity\scratch\my-workspace\CardiogastroVest\EGG DATA 4'
OUT_IMG   = r'c:\Users\kanhaiya panjiyar\.gemini\antigravity\scratch\my-workspace\CardiogastroVest\egg4_ch2_ch4_deep.png'
FS = 250

print("="*70)
print("EGG DATA 4 — Deep Channel 2 & Channel 4 Analysis")
print("="*70)

# ── LOAD & SKIP ──────────────────────────────────────────────────────────────
raw = np.loadtxt(DATA_FILE)
N_total = raw.shape[0]
skip = int(120 * FS)
data_uv = raw[skip:] * 1e6
N = len(data_uv)
t = np.arange(N) / FS
dur = N / FS
print(f"Loaded: {N_total} samples | Skipped 2 min | Analyzing {dur/60:.1f} min")

# ── FILTER (SOS stable) ─────────────────────────────────────────────────────
sos = butter(4, [0.016/(FS/2), 0.25/(FS/2)], btype='bandpass', output='sos')

channels = {
    'CH2': sosfiltfilt(sos, data_uv[:, 1]),
    'CH4': sosfiltfilt(sos, data_uv[:, 3]),
}

for name, sig in channels.items():
    print(f"  {name}: std={sig.std():.2f} µV, Vpp={sig.max()-sig.min():.1f} µV")

# ── HELPER: multi-resolution epoch PSD ───────────────────────────────────────
def multi_res_psd(sig, epoch_lengths_s=[120, 180, 240]):
    """Compute epoch-averaged PSD at multiple resolutions."""
    all_results = {}
    for ep_s in epoch_lengths_s:
        ep_n = int(ep_s * FS)
        n_ep = N // ep_n
        if n_ep < 1: continue
        psds, doms = [], []
        for e in range(n_ep):
            seg = sig[e*ep_n : (e+1)*ep_n].copy()
            # Linear detrend
            x = np.arange(len(seg))
            c = np.polyfit(x, seg, 1)
            seg -= np.polyval(c, x)
            f, p = welch(seg, FS, nperseg=ep_n, window='hann', scaling='density')
            psds.append(p)
            cpm = f * 60
            b = (cpm >= 1.0) & (cpm <= 9.0)
            if b.any():
                doms.append(cpm[b][np.argmax(p[b])])
        all_results[ep_s] = {
            'freqs': f, 'cpm': f*60,
            'avg_psd': np.mean(psds, axis=0),
            'psds': psds,
            'doms': doms, 'n_ep': n_ep
        }
    return all_results

# ── HELPER: band powers ─────────────────────────────────────────────────────
def band_powers(cpm, psd):
    brady_m = (cpm >= 0.5) & (cpm <  2.0)
    norm_m  = (cpm >= 2.0) & (cpm <= 4.0)
    tachy_m = (cpm >  4.0) & (cpm <= 9.0)
    p_b = scipy.integrate.trapezoid(psd[brady_m], cpm[brady_m]) if brady_m.any() else 0
    p_n = scipy.integrate.trapezoid(psd[norm_m],  cpm[norm_m])  if norm_m.any() else 0
    p_t = scipy.integrate.trapezoid(psd[tachy_m], cpm[tachy_m]) if tachy_m.any() else 0
    total = p_b + p_n + p_t
    return {
        'brady_pct': p_b/total*100 if total else 0,
        'norm_pct':  p_n/total*100 if total else 0,
        'tachy_pct': p_t/total*100 if total else 0,
    }

# ── HELPER: parabolic peak interpolation ─────────────────────────────────────
def para_peak(freqs, psd, lo_cpm=1.0, hi_cpm=9.0):
    cpm = freqs * 60
    band = (cpm >= lo_cpm) & (cpm <= hi_cpm)
    bi = np.where(band)[0]
    pl = np.argmax(psd[band])
    pk = bi[pl]
    if 0 < pk < len(psd)-1:
        a, b, g = psd[pk-1], psd[pk], psd[pk+1]
        d = 0.5*(a - g)/(a - 2*b + g + 1e-30)
        return (freqs[pk] + d*(freqs[1]-freqs[0])) * 60
    return cpm[pk]

# ── ANALYZE EACH CHANNEL ─────────────────────────────────────────────────────
ch_analysis = {}
for name, sig in channels.items():
    print(f"\n{'─'*70}")
    print(f"  DEEP ANALYSIS: {name}")
    print(f"{'─'*70}")

    # Multi-resolution PSD
    mr = multi_res_psd(sig)

    # Use 3-min epoch as primary
    primary = mr[180]
    dom = para_peak(primary['freqs'], primary['avg_psd'])
    bp = band_powers(primary['cpm'], primary['avg_psd'])
    dfic = np.std(primary['doms']) if len(primary['doms']) > 1 else 999

    print(f"  Dominant Freq (3-min avg): {dom:.2f} cpm")
    print(f"  Normal (2-4 cpm)         : {bp['norm_pct']:.1f}%")
    print(f"  Bradygastria (<2 cpm)    : {bp['brady_pct']:.1f}%")
    print(f"  Tachygastria (4-9 cpm)   : {bp['tachy_pct']:.1f}%")
    print(f"  DFIC (3-min epochs)      : {dfic:.3f} cpm")

    # Cross-validate with 2-min and 4-min
    for ep_s in [120, 240]:
        if ep_s in mr:
            d2 = para_peak(mr[ep_s]['freqs'], mr[ep_s]['avg_psd'])
            b2 = band_powers(mr[ep_s]['cpm'], mr[ep_s]['avg_psd'])
            print(f"  Cross-val ({ep_s//60}-min epochs): Dom={d2:.2f} cpm | Normal={b2['norm_pct']:.1f}%")

    # Wave-by-wave analysis
    wave_pks, props = find_peaks(sig, distance=int(15*FS),
                                 prominence=0.3*sig.std())
    if len(wave_pks) > 1:
        intervals = np.diff(wave_pks) / FS  # seconds
        wave_cpms = 60.0 / intervals
        amps = props['prominences']
        print(f"\n  Wave-by-wave stats ({len(wave_pks)} waves):")
        print(f"    Mean interval       : {intervals.mean():.1f} ± {intervals.std():.1f} s")
        print(f"    Mean wave rate      : {wave_cpms.mean():.2f} ± {wave_cpms.std():.2f} cpm")
        print(f"    Median wave rate    : {np.median(wave_cpms):.2f} cpm")
        print(f"    Mean amplitude      : {amps.mean():.1f} ± {amps.std():.1f} µV")
        print(f"    Max amplitude       : {amps.max():.1f} µV")
        # Fraction of waves in normal range
        normal_waves = np.sum((wave_cpms >= 2.0) & (wave_cpms <= 4.0))
        print(f"    Waves in 2-4 cpm    : {normal_waves}/{len(wave_cpms)} ({normal_waves/len(wave_cpms)*100:.0f}%)")
    else:
        wave_cpms = np.array([])
        amps = np.array([])
        intervals = np.array([])
        print(f"\n  Wave-by-wave: Insufficient peaks ({len(wave_pks)})")

    ch_analysis[name] = {
        'sig': sig, 'mr': mr, 'dom': dom, 'bp': bp, 'dfic': dfic,
        'wave_pks': wave_pks, 'wave_cpms': wave_cpms,
        'intervals': intervals, 'amps': amps if len(wave_pks)>1 else np.array([]),
        'primary': primary
    }

# Cross-channel coherence
print(f"\n{'─'*70}")
print(f"  CROSS-CHANNEL ANALYSIS: CH2 vs CH4")
print(f"{'─'*70}")
f_coh, Cxy = coherence(channels['CH2'], channels['CH4'], FS, nperseg=int(180*FS))
cpm_coh = f_coh * 60
norm_band = (cpm_coh >= 2.0) & (cpm_coh <= 4.0)
if norm_band.any():
    mean_coh = np.mean(Cxy[norm_band])
    peak_coh = np.max(Cxy[norm_band])
    print(f"  Mean coherence (2-4 cpm) : {mean_coh:.3f}")
    print(f"  Peak coherence (2-4 cpm) : {peak_coh:.3f}")
else:
    mean_coh, peak_coh = 0, 0

corr = np.corrcoef(channels['CH2'], channels['CH4'])[0, 1]
print(f"  Pearson correlation       : {corr:.4f}")

# ── PLOT ──────────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("Generating Deep CH2 & CH4 Report Plot ...")
print(f"{'='*70}")

COLORS = {'CH2': '#00e5ff', 'CH4': '#ce93d8'}

fig = plt.figure(figsize=(24, 28))
fig.patch.set_facecolor('#0b0d12')
fig.suptitle(
    'CardiogastroVest  –  EGG DATA 4  |  Deep CH2 & CH4 Analysis\n'
    'sosfiltfilt 0.016–0.25 Hz  |  Multi-Resolution PSD  |  Wave-by-Wave  |  Coherence',
    fontsize=15, fontweight='bold', color='#00e5ff', y=0.99)

gs = gridspec.GridSpec(5, 2, figure=fig, hspace=0.42, wspace=0.28,
                       top=0.95, bottom=0.04)

def style_ax(ax, title, xlabel='', ylabel='', tc='#00e5ff'):
    ax.set_facecolor('#12151c')
    ax.tick_params(colors='#aaaaaa', labelsize=8.5)
    ax.xaxis.label.set_color('#aaaaaa')
    ax.yaxis.label.set_color('#aaaaaa')
    ax.set_title(title, color=tc, fontsize=10, fontweight='bold', pad=6)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    for sp in ax.spines.values(): sp.set_edgecolor('#2a2f3e')
    ax.grid(True, alpha=0.18, color='#2a2f3e')

# ── ROW 0: Time series ──────────────────────────────────────────────────────
for j, name in enumerate(['CH2', 'CH4']):
    ax = fig.add_subplot(gs[0, j])
    col = COLORS[name]
    ca = ch_analysis[name]
    ds = 10
    tx = t[::ds]/60 + 2.0
    ax.plot(tx, ca['sig'][::ds], color=col, lw=0.8, alpha=0.9)
    if len(ca['wave_pks']):
        wp = ca['wave_pks']
        ax.plot((t[wp]/60)+2.0, ca['sig'][wp], 'o',
                color='#00ff88', ms=4, zorder=5, alpha=0.8,
                label=f"Waves ({len(wp)})")
        ax.legend(fontsize=8, facecolor='#1a1d26', edgecolor='#444', labelcolor='white')
    style_ax(ax, f"{name} — Filtered Signal  |  Dom {ca['dom']:.2f} cpm",
             xlabel='Time (min)', ylabel='Amplitude (µV)', tc=col)
    ax.set_xlim(2, N_total/FS/60)

# ── ROW 1: Multi-resolution PSD ─────────────────────────────────────────────
for j, name in enumerate(['CH2', 'CH4']):
    ax = fig.add_subplot(gs[1, j])
    col = COLORS[name]
    ca = ch_analysis[name]
    for ep_s, res in sorted(ca['mr'].items()):
        valid = res['cpm'] <= 10
        lbl = f"{ep_s//60}-min ({res['n_ep']} ep)"
        lw = 2.0 if ep_s == 180 else 1.0
        al = 1.0 if ep_s == 180 else 0.5
        ax.plot(res['cpm'][valid], res['avg_psd'][valid], lw=lw, alpha=al, label=lbl)
    ax.axvspan(2.0, 4.0, alpha=0.2, color='#00ff88')
    ax.axvline(ca['dom'], color='white', ls='--', lw=1.5, label=f"Peak {ca['dom']:.2f}")
    ax.set_xlim(0, 10)
    ax.legend(fontsize=7.5, facecolor='#1a1d26', edgecolor='#444', labelcolor='white')
    style_ax(ax, f"{name} — Multi-Resolution PSD  |  N={ca['bp']['norm_pct']:.0f}%  B={ca['bp']['brady_pct']:.0f}%  T={ca['bp']['tachy_pct']:.0f}%",
             xlabel='Frequency (cpm)', ylabel='PSD (µV²/Hz)', tc=col)

# ── ROW 2: STFT Spectrograms ────────────────────────────────────────────────
for j, name in enumerate(['CH2', 'CH4']):
    ax = fig.add_subplot(gs[2, j])
    col = COLORS[name]
    ca = ch_analysis[name]
    nperseg_stft = int(120 * FS)  # 2-min windows
    noverlap = nperseg_stft // 2
    f_stft, t_stft, Sxx = spectrogram(ca['sig'], FS,
                                       nperseg=nperseg_stft,
                                       noverlap=noverlap,
                                       window='hann')
    cpm_stft = f_stft * 60
    valid = cpm_stft <= 10.0
    ax.pcolormesh((t_stft/60)+2.0, cpm_stft[valid], Sxx[valid, :],
                  shading='gouraud', cmap='magma')
    ax.axhline(2.0, color='#00ff88', ls='--', lw=0.8, alpha=0.6)
    ax.axhline(4.0, color='#00ff88', ls='--', lw=0.8, alpha=0.6)
    ax.set_ylim(0, 10)
    style_ax(ax, f"{name} — Time-Frequency Spectrogram (2-min STFT)",
             xlabel='Time (min)', ylabel='Frequency (cpm)', tc=col)

# ── ROW 3: Wave-by-wave histograms ──────────────────────────────────────────
for j, name in enumerate(['CH2', 'CH4']):
    ax = fig.add_subplot(gs[3, j])
    col = COLORS[name]
    ca = ch_analysis[name]
    if len(ca['wave_cpms']) > 2:
        bins = np.arange(0, 10.5, 0.5)
        ax.hist(ca['wave_cpms'], bins=bins, color=col, alpha=0.7, edgecolor='white', lw=0.5)
        ax.axvspan(2.0, 4.0, alpha=0.2, color='#00ff88', label='Normal range')
        ax.axvline(np.median(ca['wave_cpms']), color='white', ls='--', lw=1.5,
                   label=f"Median {np.median(ca['wave_cpms']):.2f} cpm")
        in_norm = np.sum((ca['wave_cpms']>=2)&(ca['wave_cpms']<=4))
        ax.legend(fontsize=8, facecolor='#1a1d26', edgecolor='#444', labelcolor='white')
        style_ax(ax, f"{name} — Wave Rate Distribution  |  {in_norm}/{len(ca['wave_cpms'])} normal",
                 xlabel='Wave Rate (cpm)', ylabel='Count', tc=col)
    else:
        ax.text(0.5, 0.5, 'Insufficient waves', transform=ax.transAxes,
                ha='center', va='center', color='#aaaaaa', fontsize=14)
        style_ax(ax, f"{name} — Wave Rate Distribution", tc=col)

# ── ROW 4: Cross-channel coherence + Epoch dom freq ─────────────────────────
# Left: Coherence
ax = fig.add_subplot(gs[4, 0])
valid_coh = cpm_coh <= 10
ax.plot(cpm_coh[valid_coh], Cxy[valid_coh], color='#ffd740', lw=2.0)
ax.fill_between(cpm_coh[valid_coh], Cxy[valid_coh], alpha=0.15, color='#ffd740')
ax.axvspan(2.0, 4.0, alpha=0.2, color='#00ff88', label='Normal band')
ax.axhline(0.5, color='#ff4081', ls=':', lw=1, label='Significance')
ax.set_xlim(0, 10)
ax.set_ylim(0, 1)
ax.legend(fontsize=8, facecolor='#1a1d26', edgecolor='#444', labelcolor='white')
style_ax(ax, f"CH2 vs CH4 Coherence  |  Mean(2-4)={mean_coh:.2f}  r={corr:.3f}",
         xlabel='Frequency (cpm)', ylabel='Coherence', tc='#ffd740')

# Right: Epoch-by-epoch dominant frequency overlay
ax = fig.add_subplot(gs[4, 1])
ax.axhspan(2.0, 4.0, alpha=0.2, color='#00ff88', label='Normal 2–4 cpm')
for name in ['CH2', 'CH4']:
    ca = ch_analysis[name]
    doms = ca['primary']['doms']
    ex = np.arange(1, len(doms)+1)
    ax.plot(ex, doms, 'o-', color=COLORS[name], ms=10, lw=2.0, label=f"{name} (DFIC={ca['dfic']:.2f})")
ax.set_ylim(0, 9)
ax.set_xticks(np.arange(1, max(len(ch_analysis['CH2']['primary']['doms']),
                               len(ch_analysis['CH4']['primary']['doms']))+1))
ax.legend(fontsize=8, facecolor='#1a1d26', edgecolor='#444', labelcolor='white')
style_ax(ax, 'Epoch-by-Epoch Dominant Frequency (CH2 vs CH4)',
         xlabel='3-min Epoch #', ylabel='Dom Freq (cpm)', tc='#ffd740')

plt.savefig(OUT_IMG, dpi=130, bbox_inches='tight', facecolor='#0b0d12')
print(f"\nReport saved → {OUT_IMG}")
