from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def notch_filter(x, samplerate, plot=False):
    
    x = x - np.mean(x)
    
    high_cutoff_notch = 59 / (samplerate / 2)
    low_cutoff_notch = 61 / (samplerate / 2)

    b, a = signal.butter(4, [high_cutoff_notch, low_cutoff_notch], btype='stop')
    
    x_filt = signal.filtfilt(b, a, x.T)
         
    if plot:
        t = np.arange(0, len(x) / samplerate, 1 / samplerate)
        plt.plot(t, x)
        plt.plot(t, x_filt.T, 'k')
        plt.autoscale(tight=True)
        plt.xlabel('Time')
        plt.ylabel('Amplitude (mV)')
        plt.show()

    return x_filt


def bp_filter(x, low_f, high_f, samplerate, plot=False):
    
    # x = x - np.mean(x)
    
    high_cutoff_bp = low_f / (samplerate / 2)
    low_cutoff_bp = high_f / (samplerate / 2)

    b, a = signal.butter(10, [high_cutoff_bp, low_cutoff_bp], btype='bandpass')
    
    x_filt = signal.filtfilt(b, a, x)
    
    if plot:
        t = np.arange(0, len(x) / samplerate, 1 / samplerate)
        plt.plot(t, x)
        plt.plot(t, x_filt, 'k')
        plt.autoscale(tight=True)
        plt.xlabel('Time')
        plt.ylabel('Amplitude (mV)')
        plt.show()

    return x_filt


def plot_one_ch(x, samplerate, chname, activity):

    t = np.arange(0, len(x) / samplerate, 1 / samplerate)
    plt.plot(t, x)
    plt.autoscale(tight=True)
    plt.xlabel('Time')
    plt.ylabel('Amplitude (mV)')
    plt.title(chname + ': ' + activity)
    plt.show()


def activities_segmentation(signal, num_pulses, activities, samplerate, plot=False):

    # idx = np.arange(0, len(signal) / samplerate, 1 / samplerate)

    band = 1

    on_pulse = []
    off_pulse = []

    for idx, i in enumerate(signal):

        if i == 1 and band == 1:  # onset
            band = 0
            on_pulse.append(idx)
        if i == 0 and band == 0:  # offset
            band = 1
            off_pulse.append(idx)

    activity_onset = list(off_pulse[i] for i in range(0, num_pulses-1, 2))
    activity_offset = list(on_pulse[i] for i in range(1, num_pulses, 2))

    if num_pulses != len(on_pulse):
        print('marca manual: ', str(num_pulses), 'deteccion algoritmo: ', str(len(on_pulse)))
        plot = True

    if plot:

        # idx = np.arange(0, len(signal) / samplerate, 1 / samplerate)
        plt.plot(signal)
        plt.autoscale(tight=True)
        plt.xlabel('Time')
        plt.ylabel('Amplitude (mV)')
        for a in range(0, len(activity_onset)):
            plt.text((activity_onset[a] + activity_offset[a])/2, 0.5, activities[a])
        plt.show()

    return pd.DataFrame(list(zip(activity_onset, activity_offset)), index=activities, columns=['onset', 'offset']), activities


def calculate_mean_measure(df_measure):
    values = df_measure['index'].unique()
    df_mean = []

    for value in values:
        x = df_measure[df_measure['index'] == str(value)]
        x = x.iloc[:, 1:]
        mean_measure = x.mean()
        std_measure = x.std()
        measure = pd.concat((mean_measure, std_measure), axis=1)
        df_mean.append(measure)

    multindex_names = [['0.01', '0.01', '0.1', '0.1', '1', '1', '10', '10', '100', '100', '1000', '1000'],
              ['Mean', 'STD', 'Mean', 'STD', 'Mean', 'STD', 'Mean', 'STD', 'Mean', 'STD', 'Mean', 'STD']]

    df_mean = pd.concat(df_mean, axis=1)

    return pd.DataFrame(df_mean.values, index=values, columns=multindex_names)


def select_best_svm_parameters(results):

    best_indices = np.unravel_index(np.argmax(results.values, axis=None), results.values.shape)

    values = list(results.index)

    best_gamma = values[best_indices[0]]
    best_c = values[best_indices[1]]
    best_score = results.loc[values[best_indices[0]], values[best_indices[1]]]

    print('Best gamma: ' + str(best_gamma) + '', 'Best C: ', str(best_c), '', 'Score: ', str(best_score))
    # pd.DataFrame([best_gamma, best_c, best_score], index=['Gamma', 'C', 'Score'])
    # return [best_gamma, best_c, best_score]


def multiple_dfs_to_excel(df_list, writer, sheets, spaces):
    # writer = pd.ExcelWriter(file_name, engine='xlsxwriter')
    row = 0
    for dataframe in df_list:
        dataframe.to_excel(writer, sheet_name=sheets, startrow=row, startcol=0)
        row = row + len(dataframe.index) + spaces + 1


def get_index(feature_matrix, feature_names):

    #feature_names = ['DASDV', 'ZC']
    # feature_names = [feature_names]
    # channels = ['RM', 'LM', 'RSH', 'RIH', 'LSH', 'LIH', 'OC']
    channels = ['RM', 'LM', 'RSH', 'LSH','OC']
    index = []
    for feature in feature_names:

        for channel in channels:

            aux = feature_matrix.columns.get_level_values(0) == (feature, channel)
            index.append(np.where(aux==True))

    return np.sort(np.concatenate((index), axis=1))
