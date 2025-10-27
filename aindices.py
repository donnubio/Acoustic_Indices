#from main_acoustic_indices import *
from compute_indice import *
from acoustic_index import *
from copy import deepcopy,copy
import pandas as pd
import glob
import os
import pandas as pd
from datetime import datetime
import re
from pathlib import Path




###################### ACI AcousticComplexityIndex ###################

def AcousticComplexityIndex(audio_data=None,file_name=None,
                        windowLength=512,
                        windowHop=512,
                        windowType='hamming',
                        j_bin=5, # j_bin in seconds
                        ):
    
    if audio_data is None:
        audio_data = AudioFile(file_name)

    spectro, _ = compute_spectrogram(audio_data, 
                                     windowLength=windowLength,
                                        windowHop=windowHop,
                                        scale_audio=False,
                                        square=False,
                                        windowType=windowType,
                                        centered=False,
                                        normalized=True)

    #methodToCall = globals().get('compute_ACI')
    j_bin_samples = int(j_bin * audio_data.sr / windowHop) # transform j_bin in samples
    main_value, temporal_values = compute_ACI(spectro, j_bin_samples)
    #aci = Index(index_name, temporal_values=temporal_values, main_value=main_value)
    t = np.arange(len(temporal_values)) * (j_bin/2)

    return main_value, temporal_values, t

###################### BI BioacousticIndex ###################

def BioacousticIndex(audio_data=None,file_name=None,
                        windowLength=512,
                        windowHop=256,
                        windowType='hann',
                        min_freq=2000,
                        max_freq=8000                        
                        ):
    
    if audio_data is None:
        audio_data = AudioFile(file_name)

    spectro, frequencies = compute_spectrogram(audio_data, 
                                    windowLength=windowLength,
                                    windowHop=windowHop,
                                    scale_audio=True,
                                    square=False,
                                    windowType=windowType,
                                    centered=False,
                                    normalized=False)

    min_freq_bin = int(np.argmin([abs(e - min_freq) for e in frequencies])) # min freq in samples (or bin)
    max_freq_bin = int(np.ceil(np.argmin([abs(e - max_freq) for e in frequencies]))) # max freq in samples (or bin)

    min_freq_bin = min_freq_bin - 1 # alternative value to follow the R code
    spectro_BI = 20 * np.log10(spectro/np.max(spectro))  #  Use of decibel values. Equivalent in the R code to: spec_left <- spectro(left, f = samplingrate, wl = fft_w, plot = FALSE, dB = "max0")$amp
    spectre_BI_mean = 10 * np.log10 (np.mean(10 ** (spectro_BI/10), axis=1))     # Compute the mean for each frequency (the output is a spectre). This is not exactly the mean, but it is equivalent to the R code to: return(a*log10(mean(10^(x/a))))
    spectre_BI_mean_segment =  spectre_BI_mean[min_freq_bin:max_freq_bin]   # Segment between min_freq and max_freq
    spectre_BI_mean_segment_normalized = spectre_BI_mean_segment - min(spectre_BI_mean_segment) # Normalization: set the minimum value of the frequencies to zero.
    area = np.sum(spectre_BI_mean_segment_normalized / (frequencies[1]-frequencies[0]))   # Compute the area under the spectre curve. Equivalent in the R code to: left_area <- sum(specA_left_segment_normalized * rows_width)

    return area

###################### NDSI NormalizedDifferenceSoundIndex ###################

def NormalizedDifferenceSoundIndex(audio_data=None,file_name=None,
                        windowLength=1024,
                        #windowHop=256,
                        #windowType='hann',
                        anthrophony=[1000,2000],
                        biophony=[2000,11000]                     
                        ):
    
    if audio_data is None:
        audio_data = AudioFile(file_name)

    #frequencies, pxx = signal.welch(file.sig_float, fs=file.sr, window='hamming', nperseg=windowLength, noverlap=windowLength/2, nfft=windowLength, detrend=False, return_onesided=True, scaling='density', axis=-1) # Estimate power spectral density using Welch's method
    # TODO change of detrend for apollo
    frequencies, pxx = signal.welch(audio_data.sig_float, fs=audio_data.sr, 
                                    window='hamming', 
                                    nperseg=windowLength, 
                                    noverlap=windowLength/2, 
                                    nfft=windowLength, 
                                    detrend='constant', 
                                    return_onesided=True, 
                                    scaling='density', 
                                    axis=-1) # Estimate power spectral density using Welch's method
    avgpow = pxx * frequencies[1] # use a rectangle approximation of the integral of the signal's power spectral density (PSD)
    #avgpow = avgpow / np.linalg.norm(avgpow, ord=2) # Normalization (doesn't change the NDSI values. Slightly differ from the matlab code).

    min_anthro_bin=np.argmin([abs(e - anthrophony[0]) for e in frequencies])  # min freq of anthrophony in samples (or bin) (closest bin)
    max_anthro_bin=np.argmin([abs(e - anthrophony[1]) for e in frequencies])  # max freq of anthrophony in samples (or bin)
    min_bio_bin=np.argmin([abs(e - biophony[0]) for e in frequencies])  # min freq of biophony in samples (or bin)
    max_bio_bin=np.argmin([abs(e - biophony[1]) for e in frequencies])  # max freq of biophony in samples (or bin)

    anthro = np.sum(avgpow[min_anthro_bin:max_anthro_bin])
    bio = np.sum(avgpow[min_bio_bin:max_bio_bin])

    ndsi = (bio - anthro) / (bio + anthro)

    return ndsi

###################### ADI AcousticDiversityIndex ###################

def AcousticDiversityIndex(audio_data=None,file_name=None,
                        # windowLength=512,
                        # windowHop=512,
                        windowType='hann',
                        # freq_band_Hz,  
                        max_freq=10000, 
                        db_threshold=-50, 
                        freq_step=1000
                        ):
    
    if audio_data is None:
        audio_data = AudioFile(file_name)

    freq_band_Hz = max_freq / freq_step
    windowLength = int(audio_data.sr / freq_band_Hz)
    spectro, _ = compute_spectrogram(audio_data, 
                                    windowLength=windowLength,
                                    windowHop=windowLength,
                                    scale_audio=True,
                                    square=False,
                                    windowType=windowType,
                                    centered=False,
                                    normalized=False)

    main_value = compute_ADI(spectro, freq_band_Hz,  max_freq, db_threshold, freq_step)

    return main_value

###################### AEI AcousticEvennessIndex ###################

def AcousticEvennessIndex(audio_data=None,file_name=None,
                        # windowLength=512,
                        # windowHop=512,
                        windowType='hann',
                        # freq_band_Hz,  
                        max_freq=10000, 
                        db_threshold=-50, 
                        freq_step=1000
                        ):
    
    if audio_data is None:
        audio_data = AudioFile(file_name)

    freq_band_Hz = max_freq / freq_step
    windowLength = int(audio_data.sr / freq_band_Hz)
    spectro, _ = compute_spectrogram(audio_data, 
                                    windowLength=windowLength,
                                    windowHop=windowLength,
                                    scale_audio=True,
                                    square=False,
                                    windowType=windowType,
                                    centered=False,
                                    normalized=False)

    main_value = compute_AEI(spectro, freq_band_Hz,  max_freq, db_threshold, freq_step)

    return main_value


#################################################################################
############################### tools        ####################################
#################################################################################


def ApplyAcousticIndexToChunks(aindex_fun, 
                               audio_data=None, 
                               file_name=None, 
                               chunk_lng_sec=60, 
                               out_args_type='main_temporal',# 'main'
                               **kwargs):

    if audio_data is None:
        audio_data = AudioFile(file_name)

    chunk=deepcopy(audio_data)
    main_value_chanks=[];  temporal_values_clusters=[]; t_clusters=[]
    t_chunks = np.arange(0, audio_data.duration , chunk_lng_sec)
    for t1 in np.arange(0, audio_data.duration , chunk_lng_sec):

            #t1,t2=t-chunk_lng_sec,t+chunk_lng_sec
            t2 = t1 + chunk_lng_sec

            i1,i2=int(t1*audio_data.sr),int(t2*audio_data.sr)
            if i1<0:
                i0=0
                t1=0
            if i2>len(audio_data.sig_int):
                i2=len(audio_data.sig_int)
                t2=audio_data.duration
            # if dbg: print("t: ",t1,t2,i1,i2)
            # print("t: ",t1,t2,i1,i2,audio_data.duration)
            chunk.sig_int = audio_data.sig_int[i1:i2]
            chunk.sig_float = audio_data.sig_float[i1:i2]
            chunk.duration = len(chunk.sig_int)/float(chunk.sr)
            chunk.indices = dict()  # empty dictionary of Index    

            main_value, temporal_values, t = None, None, None
            if out_args_type=='main_temporal':
                main_value, temporal_values, t = aindex_fun( chunk, file_name=None, **kwargs)
            if out_args_type=='main':
                main_value = aindex_fun( chunk, file_name=None, **kwargs)                
            #print(main_value)
            main_value_chanks.append(main_value)
            temporal_values_clusters.append(temporal_values)
            t_clusters.append(t)

    t_centre_chunks = t_chunks + chunk_lng_sec/2            

    return main_value_chanks, temporal_values_clusters, t_clusters, t_centre_chunks         

##############################

def get_date_time_from_filename(filename):
    # convert filename "*%Y%m%d_%H%M%S*.ext" to datetime object
    str = Path(filename).stem
    d_t_str = re.search(r'(\d{8}_\d{6})', str).group(1)
    d_t = datetime.strptime(d_t_str, "%Y%m%d_%H%M%S")
    return d_t

##############################

def AcousticIndicesBanch(input_folder,
                        extension="*.flac",
                        chunk_lng_sec = 60):
    
    '''
    For audio files in the folder `input_folder` with extension `extension`, 
    it calculates Acoustic Indices with default parameters. 
    Each file is split into chunks with a length of `chunk_lng_sec`. 
    Acoustic indices are calculated for each chunk and then averaged.
    Return pandas dataframe with indices.

    '''
    


    files = glob.glob(os.path.join(input_folder, extension))
    data = []

    for file in files:

        datt = get_date_time_from_filename(file)

        audio_data = AudioFile(file)
        aci_main_val_chunks,temporal_val_clust,t_clust,t_chunks = ApplyAcousticIndexToChunks(
                                                AcousticComplexityIndex, 
                                                audio_data, file_name=None,
                                                chunk_lng_sec=chunk_lng_sec)
        adi_main_val_chunks,temporal_val_clust,t_clust,t_chunks = ApplyAcousticIndexToChunks(
                                                AcousticDiversityIndex, 
                                                audio_data, file_name=None,
                                                chunk_lng_sec=chunk_lng_sec,
                                                out_args_type='main')    
        aei_main_val_chunks,temporal_val_clust,t_clust,t_chunks = ApplyAcousticIndexToChunks(
                                                AcousticEvennessIndex, 
                                                audio_data, file_name=None,
                                                chunk_lng_sec=chunk_lng_sec,
                                                out_args_type='main')  
        bi_main_val_chunks,temporal_val_clust,t_clust,t_chunks = ApplyAcousticIndexToChunks(
                                                BioacousticIndex, 
                                                audio_data, file_name=None,
                                                chunk_lng_sec=chunk_lng_sec,
                                                out_args_type='main')  
        ndsi_main_val_chunks,temporal_val_clust,t_clust,t_chunks = ApplyAcousticIndexToChunks(
                                                NormalizedDifferenceSoundIndex, 
                                                audio_data, file_name=None,
                                                chunk_lng_sec=chunk_lng_sec,
                                                out_args_type='main')          
        
        data.append({
            'datetime': datt,
            'time': datt.hour + datt.minute/60,
            'ACI_mn': np.mean(aci_main_val_chunks),
            'ADI_mn': np.mean(adi_main_val_chunks),
            'AEI_mn': np.mean(aei_main_val_chunks),
            'BI_mn': np.mean(bi_main_val_chunks),
            'NDSI_mn': np.mean(ndsi_main_val_chunks),        
            'ACI_sd': np.std(aci_main_val_chunks),
            'ADI_sd': np.std(adi_main_val_chunks),
            'AEI_sd': np.std(aei_main_val_chunks),
            'BI_sd': np.std(bi_main_val_chunks),
            'NDSI_sd': np.std(ndsi_main_val_chunks),        
            'ACI_md': np.median(aci_main_val_chunks),
            'ADI_md': np.median(adi_main_val_chunks),
            'AEI_md': np.median(aei_main_val_chunks),
            'BI_md': np.median(bi_main_val_chunks),
            'NDSI_md': np.median(ndsi_main_val_chunks),    
            'ACI_q': np.quantile(aci_main_val_chunks, [0.25, 0.75]),
            'ADI_q': np.quantile(adi_main_val_chunks, [0.25, 0.75]),
            'AEI_q': np.quantile(aei_main_val_chunks, [0.25, 0.75]),
            'BI_q': np.quantile(bi_main_val_chunks, [0.25, 0.75]),
            'NDSI_q': np.quantile(ndsi_main_val_chunks, [0.25, 0.75])                           
        })
        
    df = pd.DataFrame(data)

    return df
