from audioseal import AudioSeal
import torchaudio
import torch
import matplotlib.pyplot as plt
import numpy as np

import matplotlib

matplotlib.use("TkAgg")


# Run this line if you wanna see the plots
plt.close = plt.show


def load_model(model_name):
    model = AudioSeal.load_generator(model_name)
    return model


def watermark_audios(path, model):
    if type(path) != list and type(path) != tuple:
        path = [path]

    watermarked_audios = []
    for audio_path in path:
        # Load audio file
        wav, sr = torchaudio.load(audio_path)

        # Ensure the audio is at the sample rate expected by the model
        if sr != 16000:
            transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            wav = transform(wav)
            sr = 16000

        # Convert to the required shape (batch, channels, samples)
        wav = wav.unsqueeze(0)  # Add batch dimension

        watermark = model.get_watermark(wav, sr)
        watermarked_audio = wav + watermark
        watermarked_audios.append(watermarked_audio)

    return watermarked_audios


def detect_watermarked_audios(path, model):
    if type(path) != list and type(path) != tuple:
        path = [path]

    results = []
    for audio_path in path:
        # Load audio file
        wav, sr = torchaudio.load(audio_path)

        # Ensure the audio is at the sample rate expected by the model
        if sr != 16000:
            transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            if wav.shape[0] > 1:
                wav = torch.mean(wav, dim=0, keepdim=True)
            wav = transform(wav)
            sr = 16000

        # Convert to the required shape (batch, channels, samples)
        wav = wav.unsqueeze(0)  # Add batch dimension

        result, message = model.detect_watermark(wav, sr)
        results.append((result, message))

    return results


# model name corresponds to the YAML card file name found in audioseal/cards
model = AudioSeal.load_generator("audioseal_wm_16bits")

# Other way is to load directly from the checkpoint
# model =  Watermarker.from_pretrained(checkpoint_path, device = wav.device)

# a torch tensor of shape (batch, channels, samples) and a sample rate
# It is important to process the audio to the same sample rate as the model
# expects. In our case, we support 16khz audio
audio_path = "./MiłyPan - Małolatki (Official Video).opus"
import os

watermarked_audio_path = os.path.splitext(audio_path)[0] + "_watermarked.wav"
wav, sr = torchaudio.load(audio_path)

# Ensure the audio is at the sample rate expected by the model
if sr != 16000:
    transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
    wavs_per_channel = []
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    # for channel in range(wav.shape[0]):
    #     wavs_per_channel.append(transform(wav[channel, :].unsqueeze(0)))
    # wav = torch.cat(wavs_per_channel)
    sr = 16000

wav = wav.unsqueeze(0)

watermark = model.get_watermark(wav, sr)

# Optional: you can add a 16-bit message to embed in the watermark
# msg = torch.randint(0, 2, (wav.shape(0), model.msg_processor.nbits), device=wav.device)
# watermark = model.get_watermark(wav, message = msg)

watermarked_audio = wav + watermark

detector = AudioSeal.load_detector("audioseal_detector_16bits")

# To detect the messages in the high-level.
result, message = detector.detect_watermark(watermarked_audio, sr)

print(
    result
)  # result is a float number indicating the probability of the audio being watermarked,
print(message)  # message is a binary vector of 16 bits


# To detect the messages in the low-level.
result, message = detector(watermarked_audio, sr)
result_2, message_2 = detector(wav, sr)

wav = wav.squeeze(0)
watermarked_audio = watermarked_audio.squeeze(0)

# Save the watermarked audio and the original audio
print(watermarked_audio.shape)
# torchaudio.save(watermarked_audio_path, watermarked_audio, sr)

# result is a tensor of size batch x 2 x frames, indicating the probability (positive and negative) of watermarking for each frame
# A watermarked audio should have result[:, 1, :] > 0.5
print(result[:, 1, :])

plt.style.use("seaborn-v0_8-whitegrid")

# plt.figure(figsize=(12, 6), dpi=150)
# plt.plot(
#     result[:, 1, :].cpu().detach().numpy(),
#     label="Watermarked",
#     color="crimson",
#     alpha=0.7,
#     linewidth=1.2,
# )
# plt.plot(
#     result_2[:, 1, :].cpu().detach().numpy(),
#     label="Original",
#     color="navy",
#     alpha=0.8,
#     linewidth=1.5,
# )
# plt.title("Watermarking Detection: Original vs Watermarked Audio", fontsize=14, pad=20)
# plt.xlabel("Frame Index", fontsize=12)
# plt.ylabel("Probability of watermark", fontsize=12)
# plt.legend(frameon=True, facecolor="white")
# plt.tight_layout()
# plt.savefig("watermarking-detection.png", bbox_inches="tight")
# plt.close()


# Waveform comparison plot
plt.figure(figsize=(12, 6), dpi=150)
plt.plot(
    wav[0, :].cpu().detach().numpy(),
    label="Original",
    color="navy",
    alpha=0.8,
    linewidth=1.5,
)
plt.plot(
    watermarked_audio[0, :].cpu().detach().numpy(),
    label="Watermarked",
    color="crimson",
    alpha=0.7,
    linestyle="--",
    linewidth=1.2,
)
plt.title("Waveform Comparison: Original vs Watermarked Audio", fontsize=14, pad=20)
plt.xlabel("Sample Index", fontsize=12)
plt.ylabel("Amplitude", fontsize=12)
plt.legend(frameon=True, facecolor="white")
plt.tight_layout()
plt.savefig("waveform-comparison.png", bbox_inches="tight")
plt.close()

# Difference plot
plt.figure(figsize=(12, 4), dpi=150)
difference = (
    wav[0, :].cpu().detach().numpy() - watermarked_audio[0, :].cpu().detach().numpy()
)
plt.plot(difference, color="forestgreen", label="Difference", alpha=0.8, linewidth=1.2)
plt.title("Waveform Difference: Original - Watermarked", fontsize=14, pad=20)
plt.xlabel("Sample Index", fontsize=12)
plt.ylabel("Amplitude Difference", fontsize=12)
plt.legend(frameon=True, facecolor="white")
plt.tight_layout()
plt.savefig("waveform-difference.png", bbox_inches="tight")
plt.close()

# Spectrogram comparison
fig, axs = plt.subplots(2, 1, figsize=(12, 8), dpi=150, sharex=True, sharey=True)

# Original spectrogram
im0 = axs[0].specgram(
    wav[0, :].cpu().detach().numpy(), Fs=sr, cmap="viridis", aspect="auto"
)
axs[0].set_title("Original Audio Spectrogram", fontsize=12)
axs[0].set_ylabel("Frequency (Hz)", fontsize=10)

# Watermarked spectrogram
im1 = axs[1].specgram(
    watermarked_audio[0, :].cpu().detach().numpy(),
    Fs=sr,
    cmap="viridis",
    aspect="auto",
)
axs[1].set_title("Watermarked Audio Spectrogram", fontsize=12)
axs[1].set_xlabel("Time (s)", fontsize=10)
axs[1].set_ylabel("Frequency (Hz)", fontsize=10)

# Add colorbar
cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
fig.colorbar(im0[3], cax=cax, label="Intensity (dB)")

plt.suptitle("Spectrogram Comparison", fontsize=14, y=0.98)
plt.tight_layout(rect=[0, 0, 0.9, 0.96])
plt.savefig("spectrogram-comparison.png", bbox_inches="tight")
plt.close()

# Create the difference between specograms


from scipy.signal import spectrogram

# Extract numpy arrays from your tensors
original_signal = wav[0, :].cpu().detach().numpy()
watermarked_signal = watermarked_audio[0, :].cpu().detach().numpy()

# Define spectrogram parameters (you can adjust these as needed)
nperseg = 256
noverlap = 128
fs = sr  # Sampling rate

# Compute spectrograms for both signals
f_orig, t_orig, Sxx_orig = spectrogram(
    original_signal, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap
)
f_water, t_water, Sxx_water = spectrogram(
    watermarked_signal, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap
)

# Convert power spectrograms to decibel (dB) scale for better visual comparison.
# Adding a small constant (e.g., 1e-10) avoids log(0) issues.
Sxx_orig_dB = 10 * np.log10(Sxx_orig + 1e-10)
Sxx_water_dB = 10 * np.log10(Sxx_water + 1e-10)

# Compute the difference (Original - Watermarked) in dB
spectrogram_difference = Sxx_orig_dB - Sxx_water_dB

# Plot the spectrogram difference
plt.figure(figsize=(12, 6), dpi=150)
mesh = plt.pcolormesh(
    t_orig, f_orig, spectrogram_difference, shading="gouraud", cmap="coolwarm"
)
plt.colorbar(mesh, label="Difference (dB)")
plt.title("Spectrogram Difference: Original - Watermarked", fontsize=14)
plt.xlabel("Time (s)", fontsize=12)
plt.ylabel("Frequency (Hz)", fontsize=12)
plt.tight_layout()
plt.savefig("spectrogram-difference.png", bbox_inches="tight")
plt.close()


# Message is a tensor of size batch x 16, indicating of the probability of each bit to be 1.
# message will be a random tensor if the detector detects no watermarking from the audio
# print(message)
