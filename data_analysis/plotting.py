import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import librosa

def plot_distribution_per_class(
    distribution: dict, nrows: int, ncols: int, figsize: tuple[int, int],
    x_label: str, y_label: str, label_decoding: dict = None, grid: bool = True,
    plot_type: str = 'barplot') -> None:
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()  # Flatten the axes array for easy indexing
    
    labels = list(distribution.keys())
    count = 0
    
    for i in range(nrows):
        for j in range(ncols):
            if count >= len(labels):
                break  # Exit loop if no more labels
            
            ax = axes[count]
            label = labels[count]
            title = label_decoding[label] if label_decoding else label
            try:
                x = list(distribution[label].keys())
                y = list(distribution[label].values())
            except:
                x = distribution[label]
            
            if plot_type == 'barplot':
                sns.barplot(ax=ax, x=x, y=y)
            elif plot_type == 'log_histogram':
                ax.hist(x, weights=y, bins=50, log=True)
            elif plot_type == 'cdf':
                data_sorted = np.sort(x)
                cdf = np.cumsum(y) / np.sum(y)
                ax.plot(data_sorted, cdf, marker='.', linestyle='none')
            elif plot_type == 'boxplot':
                sns.boxplot(ax=ax, x=x)
            elif plot_type == 'violinplot':
                sns.violinplot(ax=ax, x=x)
            elif plot_type == 'kde':
                sns.kdeplot(ax=ax, x=x, weights=y)
            
            ax.set_title(title, fontweight='bold', fontsize=16)
            ax.grid(grid)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            
            count += 1
    
    plt.tight_layout()
    plt.show()


def plot_power_spectrum_log(audio_path:str|np.ndarray,sr_cont=None,colour="blue",
                   title="Amplitude Response")->None:
    if sr_cont:
        y_cont = audio_path
    else:
        y_cont, sr_cont = librosa.load(audio_path)
    fft_result = np.fft.fft(y_cont)
    amplitude_spectrum = np.log10(np.abs(fft_result))
    frequencies = np.fft.fftfreq(len(amplitude_spectrum), 1/sr_cont)
    positive_frequencies = frequencies[:len(frequencies)//2]
    positive_amplitude_spectrum = amplitude_spectrum[:len(amplitude_spectrum)//2]
    plt.figure(figsize=(20, 5))
    plt.xlabel('Frequency (Hz)', fontsize=14)
    plt.ylabel('Amplitude', fontsize=14)
    plt.title(title, fontsize=16)
    # plt.yscale('log')  
    plt.xlim(0, max(positive_frequencies))  
    plt.ylim(1e-2, max(positive_amplitude_spectrum))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.axhline(y=1, color='k', linestyle='--', linewidth=0.7)
    plt.axvline(x=0, color='k', linestyle='--', linewidth=0.7)

    plt.fill_between(positive_frequencies, positive_amplitude_spectrum, color=colour, alpha=0.5)

    peak_freq = positive_frequencies[np.argmax(positive_amplitude_spectrum)]
    peak_amp = np.max(positive_amplitude_spectrum)
    plt.annotate(
        f'Peak Frequency: {peak_freq:.2f} Hz', xy=(peak_freq, peak_amp), 
        xytext=(peak_freq + 1000, peak_amp),arrowprops=dict(facecolor='black', shrink=0.05),
        fontsize=12, color='red')

    plt.show()

def plot_power_spectrum(signal, sampling_rate=16000, num_samples=400, title='Power Spectrum of the Signal', color='blue', fill_color='blue', annotate_color='red'):
    """
    Plot the power spectrum of a given signal.

    Parameters:
    signal (array): The input signal.
    sampling_rate (int): The sampling rate of the signal.
    num_samples (int): The number of samples in the signal.
    title (str): The title of the plot.
    color (str): The color of the power spectrum line.
    fill_color (str): The color to fill under the power spectrum.
    annotate_color (str): The color of the annotation text.
    """
    
    # Generate a time array
    t = np.linspace(0, num_samples / sampling_rate, num_samples, endpoint=False)

    # Compute the Fourier Transform of the signal
    fft_result = np.fft.fft(signal)
    fft_freq = np.fft.fftfreq(num_samples, 1 / sampling_rate)

    # Compute the power spectrum (magnitude squared of the Fourier Transform)
    power_spectrum = np.abs(fft_result) ** 2

    # Plot the power spectrum
    plt.figure(figsize=(20, 5))
    plt.plot(fft_freq[:num_samples // 2], power_spectrum[:num_samples // 2], color=color)
    plt.xlabel('Frequency (Hz)', fontsize=14)
    plt.ylabel('Power', fontsize=14)
    plt.title(title, fontsize=16)
    plt.xlim(0, max(fft_freq[:num_samples // 2]))
    plt.ylim(0, max(power_spectrum[:num_samples // 2]))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.axhline(y=1, color='k', linestyle='--', linewidth=0.7)
    plt.axvline(x=0, color='k', linestyle='--', linewidth=0.7)

    # Fill between the power spectrum for a better visual effect
    plt.fill_between(fft_freq[:num_samples // 2], power_spectrum[:num_samples // 2], color=fill_color, alpha=0.5)

    # Annotate the peak frequency
    peak_freq = fft_freq[:num_samples // 2][np.argmax(power_spectrum[:num_samples // 2])]
    peak_power = np.max(power_spectrum[:num_samples // 2])
    plt.annotate(
        f'Peak Frequency: {peak_freq:.2f} Hz', xy=(peak_freq, peak_power), 
        xytext=(peak_freq + 1000, peak_power), arrowprops=dict(facecolor='black', shrink=0.05),
        fontsize=12, color=annotate_color)

    plt.show()