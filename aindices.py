#from main_acoustic_indices import *
from compute_indice import *
from acoustic_index import *
from copy import deepcopy,copy


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



def ApplyAcousticIndexToChunks(aindex_fun, audio_data=None, file_name=None, chunk_lng_sec=60, **kwargs):

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

            main_value, temporal_values, t = aindex_fun( chunk, file_name=None, **kwargs)
            #print(main_value)
            main_value_chanks.append(main_value)
            temporal_values_clusters.append(temporal_values)
            t_clusters.append(t)

    t_centre_chunks = t_chunks + chunk_lng_sec/2            

    return main_value_chanks, temporal_values_clusters, t_clusters, t_centre_chunks            