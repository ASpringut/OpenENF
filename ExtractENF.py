from scipy.fft import rfft, irfft, rfftfreq
from scipy.io import wavfile
from scipy.signal import butter, sosfilt, spectrogram, get_window, stft
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np

from pydub import AudioSegment 

#mains frequency in Hz
MAINS_FREQUENCY = 60.0
FREQUENCY_BANDWITH = 1.0
MIN_FREQ = MAINS_FREQUENCY - FREQUENCY_BANDWITH/2
MAX_FREQ = MAINS_FREQUENCY + FREQUENCY_BANDWITH/2

def plotFrequencyDomain(data, samplesPerSecond):
    freqDomain = rfft(data)
    freqs = rfftfreq(data.size, 1.0/samplesPerSecond)

    plt.plot(freqs, freqDomain)
    plt.show()

def getSpectrogram(data, samplesPerSecond):
    window = get_window("boxcar", samplesPerSecond)
    f, t, Zxx = stft(data, samplesPerSecond, window=window, nperseg=samplesPerSecond)
    return f, t, Zxx

def plotSpectrogram(data, samplesPerSecond):
    f, t, Zxx = getSpectrogram(data, samplesPerSecond)
    plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.ylim(50,70)
    plt.show()

def findInterpolatedPeak(FFTFreqs, FFTBins):

    peakIndex = np.argmax(FFTBins)
    peakAmplitude = FFTBins[peakIndex]
    lowerAmplidue = FFTBins[peakIndex-1]
    upperAmplidue = FFTBins[peakIndex+1]
    peakFrequency = FFTFreqs[peakIndex]
    lowerFrequency = FFTFreqs[peakIndex-1]
    upperFrequency = FFTFreqs[peakIndex+1]

    '''
    print(f"""Index: {index}  Peak amplitude values:\
        {lowerFrequency}Hz {lowerAmplidue:.4f}\
        {peakFrequency}Hz {peakAmplitude:.4f}\
        {upperFrequency}Hz {upperAmplidue:.4f}""")
    '''
    frequencies = (lowerFrequency,peakFrequency,upperFrequency)
    amplitudes = (lowerAmplidue,peakAmplitude,upperAmplidue)

    interpolation = interp1d(frequencies,amplitudes,kind='quadratic')
    #minimize the inverse of our function to find the maximum, initial guess is peak
    negative = lambda x : -1.0*(interpolation(x))
    res = minimize(negative, frequencies[1])

    '''
    #Plot Interplation
    xnew = np.arange(frequencies[0],frequencies[2],0.001)
    ynew = f(xnew)
    plt.plot(frequencies, amplitudes, 'o', xnew, ynew, '-')
    plt.show()
    '''
    return res.x[0]


if __name__ == "__main__":
    samplesPerSecond, data = wavfile.read('audio.wav')
    data
    #time in seconds at each sample
    time = [1.0*x/samplesPerSecond for x in range(len(data))]
    samplePeriod = 1.0/samplesPerSecond

    print('Data length: ', data.size)
    print('Samples per second: ', samplesPerSecond ,' sample period: ',samplePeriod)
    print('Clip length: ', data.size/samplesPerSecond)

    '''
    Remove noise
    '''
    sos = butter(N=10, Wn =[MIN_FREQ, MAX_FREQ], btype = 'bandpass', output = 'sos', fs=samplesPerSecond)
    filteredSignal = sosfilt(sos, data)

    f,t,Zxx = getSpectrogram(filteredSignal,samplesPerSecond)

    #print(t)
    magZxx = np.abs(Zxx)
    #swap the axes so the array contains lists of all the FFT bins at each time step
    magZxxFT = np.swapaxes(magZxx, 0,1)

    peakFrequencies = []

    #possible this will fail if the peak amplitude is ever the first or last bin
    for FFTBins in magZxxFT:
        peakFreq = findInterpolatedPeak(f, FFTBins)
        peakFrequencies.append(peakFreq)

    print(peakFrequencies)
    plt.plot(t[2:-2], peakFrequencies[2:-2], 'o')
    plt.show()








    '''
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.set_title('Filtered and unfiltered audio')
    ax1.plot(time, data)
    ax2.plot(time, filteredSignal)
    plt.show()
    '''

    #plotFrequencyDomain(filteredSignal, samplesPerSecond)
    #plotSpectrogram(filteredSignal, samplesPerSecond)