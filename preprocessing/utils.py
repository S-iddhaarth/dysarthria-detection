import numpy as np
from scipy.fftpack import dct

def pre_emphasis_filter(signal: np.ndarray, coeff: float=0.97) -> np.ndarray:
    """
    Applies a pre-emphasis filter to an audio signal.

    Parameters
    ----------
    signal : np.ndarray
        The input audio signal to be filtered. This is a 1-dimensional NumPy array containing the audio samples.
    coeff : float
        The pre-emphasis coefficient. A typical value is 0.97. This coefficient controls the amount of emphasis
        applied to the high frequencies.

    Returns
    -------
    np.ndarray
        The pre-emphasized signal as a 1-dimensional NumPy array. The output signal has the same length as the input signal.

    Notes
    -----
    - The pre-emphasis filter is used to amplify the high-frequency components of the audio signal. This can improve
      the performance of various signal processing tasks, such as feature extraction and speech recognition.
    - The filter is defined by the equation:
      
        y[n] = x[n] - coeff * x[n-1]
      
      where `x` is the input signal, `y` is the output signal, and `coeff` is the pre-emphasis coefficient.

    Examples
    --------
    >>> import numpy as np
    >>> signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> coeff = 0.97
    >>> emphasized_signal = pre_emphasis_filter(signal, coeff)
    >>> print(emphasized_signal)
    [ 1.    1.03  1.03  1.03  1.03]
    """
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])

def frame(
        signal: np.ndarray, sr: int, frame_size: float = 0.025,
        frame_stride: float = 0.01
    ) -> tuple[np.ndarray, int]:
    """
    Splits an audio signal into overlapping frames.

    Parameters
    ----------
    signal : np.ndarray
        The audio signal to be framed. This is a 1-dimensional NumPy array containing the audio samples.
    sr : int
        The sampling rate of the audio signal in Hz.
    frame_size : float, optional
        The length of each frame in seconds. Default is 0.025 (25 ms).
    frame_stride : float, optional
        The step or stride between successive frames in seconds. This defines the amount of overlap between frames.
        Default is 0.01 (10 ms).

    Returns
    -------
    tuple[np.ndarray, int]
        A tuple containing:
        - frames (np.ndarray): A 2-dimensional NumPy array where each row contains one frame of the signal. The shape 
          of the array is (num_frames, frame_length), where num_frames is the total number of frames and frame_length 
          is the number of samples in each frame.
        - frame_length (int): The length of each frame in samples.

    Notes
    -----
    - This function pads the signal with zeros if the number of samples in the signal is not sufficient to fill the 
      last frame completely.
    - The frames are created with overlap based on the frame_stride parameter. For example, if frame_size is 0.025 
      seconds (25 ms) and frame_stride is 0.01 seconds (10 ms), there will be a 15 ms overlap between consecutive frames.

    Examples
    --------
    >>> import numpy as np
    >>> sr = 16000  # Sampling rate
    >>> signal = np.arange(0, 32000)  # Example signal
    >>> frame_size = 0.025  # 25 ms
    >>> frame_stride = 0.01  # 10 ms
    >>> frames, frame_length = frame(signal, sr, frame_size, frame_stride)
    >>> print(frames.shape)
    (1998, 400)  # 1998 frames, each with 400 samples
    >>> print(frame_length)
    400  # Length of each frame in samples
    """
    frame_length, frame_step = frame_size * sr, frame_stride * sr
    signal_length = len(signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))

    pad_signal_length_c = num_frames * frame_step + frame_length
    z_c = np.zeros((pad_signal_length_c - signal_length))
    pad_signal_c = np.append(signal, z_c)

    indices_c = np.tile(
        np.arange(0, frame_length), (num_frames, 1)) + np.tile(
            np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal_c[indices_c.astype(np.int32, copy=False)]
    return frames, frame_length

def windowing(frames:np.ndarray,size:int,window:callable)->np.ndarray:
    return frames*window(size)

def fourier_transform(frames,NFFT=512):
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum
    return pow_frames

def mel_filter_bank(
    pow_frames: np.ndarray, sr: int, NFFT: int = 512,
    nfilt: int = 40, low_freq_mel: int = 0) -> np.ndarray:
    """
    Computes the Mel filter bank energies for a given power spectrum of frames.

    Parameters
    ----------
    pow_frames : np.ndarray
        A 2D array of shape (num_frames, num_fft_bins) containing the power spectrum of the frames.
    sr : int
        The sample rate of the audio signal.
    NFFT : int, optional
        The number of FFT points. Default is 512.
    nfilt : int, optional
        The number of Mel filters to use. Default is 40.
    low_freq_mel : int, optional
        The lowest frequency (in Mel) for the Mel filter bank. Default is 0.

    Returns
    -------
    np.ndarray
        A 2D array of shape (num_frames, nfilt) containing the Mel filter bank energies in dB.

    Notes
    -----
    - The Mel scale is a perceptual scale of pitches judged by listeners to be equal in distance from one another.
    - This function creates a filter bank where each filter is a triangular filter on the linear scale, but equally spaced on the Mel scale.
    - The power spectrum of the frames is multiplied by the filter bank to produce the Mel filter bank energies.
    - The resulting energies are converted to the logarithmic dB scale for better perceptual understanding.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.io import wavfile
    >>> sr, signal = wavfile.read('path_to_audio.wav')
    >>> signal = signal / np.max(np.abs(signal))  # Normalize the signal
    >>> frames = frame(signal, sr, frame_size=0.025, frame_stride=0.01)
    >>> mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    >>> pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))
    >>> filter_banks = mel_filter_bank(pow_frames, sr, NFFT, nfilt=40)
    >>> print(filter_banks.shape)
    (num_frames, 40)

    """
    high_freq_mel = 2595 * np.log10(1 + (sr / 2) / 700)  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)  # Convert Mel to Hz
    bin_c = np.floor((NFFT + 1) * hz_points / sr)

    fbank_c = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m_c in range(1, nfilt + 1):
        f_m_minus_c = int(bin_c[m_c - 1])   # left
        f_m_c = int(bin_c[m_c])             # center
        f_m_plus_c = int(bin_c[m_c + 1])    # right

        for k_c in range(f_m_minus_c, f_m_c):
            fbank_c[m_c - 1, k_c] = (k_c - bin_c[m_c - 1]) / (bin_c[m_c] - bin_c[m_c - 1])
        for k_c in range(f_m_c, f_m_plus_c):
            fbank_c[m_c - 1, k_c] = (bin_c[m_c + 1] - k_c) / (bin_c[m_c + 1] - bin_c[m_c])

    filter_banks_c = np.dot(pow_frames, fbank_c.T)
    filter_banks_c = np.where(filter_banks_c == 0, np.finfo(float).eps, filter_banks_c)  # Numerical stability
    filter_banks_c = 20 * np.log10(filter_banks_c)  # dB

    return filter_banks_c

def mfcc(filter_banks: np.ndarray, num_ceps: int = 12) -> np.ndarray:
    """
    Computes the Mel-Frequency Cepstral Coefficients (MFCCs) from the filter bank energies.

    Parameters
    ----------
    filter_banks : np.ndarray
        A 2D array of shape (num_frames, num_filters) containing the filter bank energies in dB.
    num_ceps : int, optional
        The number of cepstral coefficients to return. Default is 12.

    Returns
    -------
    np.ndarray
        A 2D array of shape (num_frames, num_ceps) containing the MFCCs.

    Notes
    -----
    - MFCCs are a compact representation of the power spectrum of an audio signal, commonly used in speech and audio processing.
    - The Discrete Cosine Transform (DCT) is applied to the log filter bank energies to decorrelate the energies and produce the cepstral coefficients.
    - Typically, the first few coefficients represent the most significant features of the signal.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.fftpack import dct
    >>> # Assuming filter_banks is already computed
    >>> filter_banks = np.random.rand(100, 40)  # Example filter bank energies
    >>> mfccs = mfcc(filter_banks, num_ceps=12)
    >>> print(mfccs.shape)
    (100, 12)

    References
    ----------
    - [1] Wikipedia: Mel-frequency cepstrum (https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)
    - [2] Davis, S. B., & Mermelstein, P. (1980). Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences. IEEE Transactions on Acoustics, Speech, and Signal Processing, 28(4), 357-366.

    """
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, :num_ceps]
    return mfcc
