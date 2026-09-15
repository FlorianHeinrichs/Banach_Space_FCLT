#
# real_data.py
#
# Project: Functional Central Limit Theorem in Banach Spaces
# Date: 2026-02-11
# Author: Florian Heinrichs
#
# Real data experiments using:
# - Climate data obtained from DWD at:
# https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes/air_temperature/historical/
# - The "Consumer-Grade EEG and Eye-Tracking Dataset" dataset from
# https://zenodo.org/records/14860668

from eeg_et_benchmark.load_data import load_dataset
import numpy as np
import pandas as pd
from scipy.signal import spectrogram, get_window

from self_normalization import self_normalization_test
from simulate_quantiles import simulate_quantile


def experiment_temperature(filepaths: dict,
                           quantile: float,
                           n_time: int = None,
                           local_time: bool = True) -> np.array:
    """
    Conduct experiment for temperature data.

    :param filepaths: Dictionary containing city names as keys and filepaths as
        values.
    :param quantile: Quantile used for testing.
    :param n_time: Length of time series (only use last n_time days). If None,
        use full time series.
    :param local_time: Indicates whether UTC is used of local time.
    :return: NumPy array of test decisions.
    """
    results = {}

    for city, fp in filepaths.items():
        data = load_temperature(fp, local_time=local_time)

        if n_time is not None:
            data = data[-n_time:]

        n_days, n_rec_per_day = data.shape
        hours = np.arange(n_rec_per_day) / n_rec_per_day * 24

        weights = np.zeros(n_rec_per_day)
        morning = np.argwhere((6 <= hours) & (hours <= 9))[:, 0]
        evening = np.argwhere((16 <= hours) & (hours <= 19))[:, 0]
        weights[morning] = - 1 / len(morning)
        weights[evening] = 1 / len(evening)
        test_decision = self_normalization_test(data, weights, quantile,
                                                return_p_value=True)

        weights2 = np.zeros(n_rec_per_day)
        daytime = np.argwhere((6 <= hours) & (hours <= 20))[:, 0]
        weights2[daytime] = 1 / len(daytime)
        test_decision2 = self_normalization_test(data, weights2, quantile,
                                                 return_p_value=True)

        results[city] = (test_decision, test_decision2)

    return results


def load_temperature(filepath: str, local_time: bool = True) -> np.array:
    """
    Loads temperature data from a specified file. Missing values are indicated
    by -999 and days containing missing values are filtered out.

    :param filepath: Path to data.
    :param local_time: Indicates whether UTC is used of local time.
    :return: NumPy array of temperature values, where each row corresponds to a
        day.
    """
    df = pd.read_csv(filepath, usecols=[0, 2])
    df.columns = ['date', 'temperature']

    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d%H%M')

    if local_time:
        df['date'] = df['date'].dt.tz_localize('Europe/Berlin',
                                               ambiguous='NaT',
                                               nonexistent='NaT')
        df.loc[df['date'].isna(), 'temperature'] = -999

    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    pivot_df = df.pivot_table(index=['year', 'month', 'day'],
                              columns=['hour', 'minute'],
                              values='temperature', fill_value=-999).to_numpy()
    data = pivot_df[~(pivot_df < -998).any(axis=1)]

    return data


def experiment_eeg(folder: str, quantile: float, task: str = None) -> np.array:
    """
    Conduct experiment for "Consumer-Grade EEG and Eye-Tracking Dataset".

    :param folder: Path to folder containing EEG and Eye-Tracking data.
    :param quantile: Quantile used for testing.
    :param task: Specify "task" from EEG dataset. Defaults to "level-2-smooth".
    :return: NumPy array of test decisions.
    """
    if task is None:
        task = "level-2-smooth"

    eeg_columns = ['EEG_TP9', 'EEG_AF7', 'EEG_AF8', 'EEG_TP10']
    exclude = ["P002_01", "P004_01"] + [
        f"P0{k}_01" for k in list(range(16, 21)) + list(range(62, 68)) + [79]]

    recordings = load_dataset(folder=folder, task=task, exclude=exclude)
    recordings = recordings[0] + recordings[1]

    results = []

    for rec in recordings:
        eeg = rec[eeg_columns].to_numpy()
        freqs, spec = calculate_spectrogram(eeg)  # (n_channels, n_freqs, n_time)
        spec = spec.reshape((-1, spec.shape[-1])).transpose()
        log_spec = np.log10(spec)

        weights = np.zeros_like(freqs)
        alpha = np.argwhere((8 <= freqs) & (freqs <= 12))[:, 0]
        beta = np.argwhere((13 <= freqs) & (freqs <= 30))[:, 0]
        weights[alpha] = - 1 / len(alpha)
        weights[beta] = 1 / len(beta)
        weights = np.concatenate([weights] * 4)

        test_decision = self_normalization_test(log_spec, weights, quantile)
        results.append(test_decision)

    return results


def calculate_spectrogram(x: np.ndarray, fs: float = 256.,
                          win_sec: float = 2.0,
                          overlap_frac: float = 0.75,
                          window_type: str = 'hann') -> tuple:
    n_samples, n_channels = x.shape
    win_len = int(round(win_sec * fs))
    noverlap = int(round(overlap_frac * win_len))

    win = get_window(window_type, win_len, fftbins=True)

    spec = []
    for ch in range(n_channels):
        freqs, t, S = spectrogram(x[:, ch], fs=fs, window=win, nperseg=win_len,
                                  noverlap=noverlap)
        spec.append(S)
    spec = np.stack(spec, axis=0)

    return freqs, spec


if __name__ == '__main__':
    alpha = 0.05
    quantile = simulate_quantile(alpha, n_repetitions=10000, n_grid=100)
    temperature_data = {"Aachen": "../data/aachen.csv",
                        "Bochum": "../data/bochum.csv",
                        "Göttingen": "../data/goettingen.csv"}

    for n_time in [None, 365]:
        print(f"Experiment - {n_time=}")

        results = experiment_temperature(temperature_data, quantile,
                                         local_time=False, n_time=n_time)

        for city, result in results.items():
            res0, res1 = result
            print(f"{city}")
            print(f"Reject Null (no change during the day): {res1}")
            print(f"Reject Null (no change between morning and evening): {res0}")

    # for level in [1, 2]:
        # for task_type in ['saccades', 'smooth']:
            # task = f"level-{level}-{task_type}"
            # print(f"Experiment - {task=}")
            # folder = "path/to/data"
            # results = experiment_eeg(folder, quantile, task=task)
            # print(f"{len(results)} EEG recordings"
                  # f" - attention drift in {sum(results)}.")