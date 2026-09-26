# CardiogastroVest

A wearable biomedical data acquisition and analysis system for simultaneous ECG (Electrocardiography) and EGG (Electrogastrography) recording.

---

## Project Structure

```
CardiogastroVest/
├── scripts/
│   ├── analyze_egg4_ch2_ch4.py   # Final EGG analysis — EGG DATA 4, CH2 & CH4 deep pipeline
│   └── analyze_ecg2.py            # Final ECG analysis
├── outputs/
│   ├── ecg/                       # ECG analysis plots & reports
│   └── egg/                       # EGG analysis plots & reports
└── README.md
```

---

## Scripts

### `scripts/analyze_egg4_ch2_ch4.py`
Deep per-channel EGG analysis for EGG DATA 4 (CH2 & CH4).  
Pipeline:
- Skip first 2 min (electrode settling)
- Bandpass filter: 0.016–0.25 Hz (sosfiltfilt, stable)
- Multi-resolution PSD: 2-min, 3-min, 4-min epoch averages
- Time-frequency spectrogram (STFT)
- Wave-by-wave analysis: inter-wave intervals, amplitude stats
- Cross-channel coherence / correlation
- Per-epoch dominant frequency stability index (DFIC)

### `scripts/analyze_ecg2.py`
ECG signal analysis from ADS1299 device output.  
Pipeline:
- Parse `Device_0_Volts.xls` (ADS1299 output)
- Notch filter (50 Hz), bandpass filter
- R-peak detection, HRV analysis
- PSD and full signal report

---

## Hardware
- **ADC**: Texas Instruments ADS1299 (8-channel, 24-bit biosignal front-end)
- **Sampling Rate**: 250 Hz
- **EGG Gain**: 24×  
- **ECG Gain**: 6×

---

## Dependencies

```bash
pip install numpy scipy matplotlib
```

---

## Data
Raw data files are stored locally and excluded from the repository (see `.gitignore`).  
EGG data: `EGG DATA 1–4` | ECG data: `DATA_ECG`, `DATA_ECG1`
