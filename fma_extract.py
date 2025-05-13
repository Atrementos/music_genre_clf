import pandas as pd
import numpy as np
import os
import librosa
import itertools
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

tracks = pd.read_csv('...', index_col=0, header=[0, 1])

col_names = [
    'stft_mean_0', 'stft_std_0',
    'stft_mean_1', 'stft_std_1',
    'stft_mean_2', 'stft_std_2',
    'stft_mean_3', 'stft_std_3',
    'stft_mean_4', 'stft_std_4',
    'stft_mean_5', 'stft_std_5',
    'stft_mean_6', 'stft_std_6',
    'stft_mean_7', 'stft_std_7',
    'stft_mean_8', 'stft_std_8',
    'stft_mean_9', 'stft_std_9',
    'stft_mean_10', 'stft_std_10',
    'stft_mean_11', 'stft_std_11',
    'cqt_mean_0', 'cqt_std_0',
    'cqt_mean_1', 'cqt_std_1',
    'cqt_mean_2', 'cqt_std_2',
    'cqt_mean_3', 'cqt_std_3',
    'cqt_mean_4', 'cqt_std_4',
    'cqt_mean_5', 'cqt_std_5',
    'cqt_mean_6', 'cqt_std_6',
    'cqt_mean_7', 'cqt_std_7',
    'cqt_mean_8', 'cqt_std_8',
    'cqt_mean_9', 'cqt_std_9',
    'cqt_mean_10', 'cqt_std_10',
    'cqt_mean_11', 'cqt_std_11',
    'cens_mean_0', 'cens_std_0',
    'cens_mean_1', 'cens_std_1',
    'cens_mean_2', 'cens_std_2',
    'cens_mean_3', 'cens_std_3',
    'cens_mean_4', 'cens_std_4',
    'cens_mean_5', 'cens_std_5',
    'cens_mean_6', 'cens_std_6',
    'cens_mean_7', 'cens_std_7',
    'cens_mean_8', 'cens_std_8',
    'cens_mean_9', 'cens_std_9',
    'cens_mean_10', 'cens_std_10',
    'cens_mean_11', 'cens_std_11',
    'mfccs_mean_0', 'mfccs_std_0',
    'mfccs_mean_1', 'mfccs_std_1',
    'mfccs_mean_2', 'mfccs_std_2',
    'mfccs_mean_3', 'mfccs_std_3',
    'mfccs_mean_4', 'mfccs_std_4',
    'mfccs_mean_5', 'mfccs_std_5',
    'mfccs_mean_6', 'mfccs_std_6',
    'mfccs_mean_7', 'mfccs_std_7',
    'mfccs_mean_8', 'mfccs_std_8',
    'mfccs_mean_9', 'mfccs_std_9',
    'mfccs_mean_10', 'mfccs_std_10',
    'mfccs_mean_11', 'mfccs_std_11',
    'mfccs_mean_12', 'mfccs_std_12',
    'mfccs_mean_13', 'mfccs_std_13',
    'mfccs_mean_14', 'mfccs_std_14',
    'mfccs_mean_15', 'mfccs_std_15',
    'mfccs_mean_16', 'mfccs_std_16',
    'mfccs_mean_17', 'mfccs_std_17',
    'mfccs_mean_18', 'mfccs_std_18',
    'mfccs_mean_19', 'mfccs_std_19',
    'rms_mean_0', 'rms_std_0',
    'centroid_mean_0', 'centroid_std_0',
    'bandwidth_mean_0', 'bandwidth_std_0',
    'contrast_mean_0', 'contrast_std_0',
    'contrast_mean_1', 'contrast_std_1',
    'contrast_mean_2', 'contrast_std_2',
    'contrast_mean_3', 'contrast_std_3',
    'contrast_mean_4', 'contrast_std_4',
    'contrast_mean_5', 'contrast_std_5',
    'contrast_mean_6', 'contrast_std_6',
    'flatness_mean_0', 'flatness_std_0',
    'rolloff_mean_0', 'rolloff_std_0',
    'tonnetz_mean_0', 'tonnetz_std_0',
    'tonnetz_mean_1', 'tonnetz_std_1',
    'tonnetz_mean_2', 'tonnetz_std_2',
    'tonnetz_mean_3', 'tonnetz_std_3',
    'tonnetz_mean_4', 'tonnetz_std_4',
    'tonnetz_mean_5', 'tonnetz_std_5',
    'zcr_mean_0', 'zcr_std_0',
    'genre', 'track_id'
]

BASE_PATH = '...'

def get_audio_paths(base_path):
    paths = []

    for audio_dir in os.listdir(base_path):
        audio_path = os.path.join(base_path, audio_dir)

        if os.path.isdir(audio_path): # Check if it is a file or directory (need directories)
            for audio_name in os.listdir(audio_path):
                audio_full_path = os.path.join(audio_path, audio_name)

                paths.append(audio_full_path)

    return paths

def flatten_feature(feature):
    return list(itertools.chain.from_iterable(
        (np.mean(f), np.std(f)) for f in feature
    ))

def mean_std(feature):
    return np.mean(feature), np.std(feature)

def extract_audio_features(path):
    try:
        y, sr = librosa.load(path, sr=22050, mono=True)

        if y is None or len(y) == 0 or np.all(y == 0):
            raise ValueError("Empty or silent audio signal")

        total_len = len(y)
        part_len = 3 * sr
        split_parts = round(total_len / part_len)
        features_list = []

        for i in range(split_parts):
            start = i * part_len
            end = start + part_len

            snippet = y[start:end]
            if len(snippet) < part_len:
                snippet = np.pad(snippet, (0, part_len - len(snippet)), mode='constant')

            if librosa.feature.rms(y=snippet).mean() < 1e-4:
                print(f"Snippet in {path} skipped.")
                continue

            stft = librosa.feature.chroma_stft(y=snippet, sr=sr)
            cqt = librosa.feature.chroma_cqt(y=snippet, sr=sr)
            cens = librosa.feature.chroma_cens(y=snippet, sr=sr)
            mfccs = librosa.feature.mfcc(y=snippet, sr=sr, n_mfcc=20)
            rms = librosa.feature.rms(y=snippet)
            centroid = librosa.feature.spectral_centroid(y=snippet, sr=sr)
            bandwidth = librosa.feature.spectral_bandwidth(y=snippet, sr=sr)
            contrast = librosa.feature.spectral_contrast(y=snippet, sr=sr)
            flatness = librosa.feature.spectral_flatness(y=snippet)
            rolloff = librosa.feature.spectral_rolloff(y=snippet, sr=sr)
            y_harmonic, _ = librosa.effects.hpss(snippet)
            tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y=snippet)
            # tempo = librosa.feature.tempo(y=snippet, sr=sr)[0]

            features_list.append([
                *flatten_feature(stft),
                *flatten_feature(cqt),
                *flatten_feature(cens),
                *flatten_feature(mfccs),
                *mean_std(rms),
                *mean_std(centroid),
                *mean_std(bandwidth),
                *flatten_feature(contrast),
                *mean_std(flatness),
                *mean_std(rolloff),
                *flatten_feature(tonnetz),
                *mean_std(zcr),
                # tempo
            ])
        
        return features_list
    except Exception as e:
        print(f"Failed on {path} with error: {type(e).__name__}: {e}")
        return []
    
def process_file(item):
    path, genre = item
    try:
        file_name = os.path.basename(path)
        track_id = int(os.path.splitext(file_name)[0])

        if pd.isna(genre):
            return []

        features_list = extract_audio_features(path)
        return [features + [genre, track_id] for features in features_list]
    
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return []

def main():
    paths = get_audio_paths(BASE_PATH)

    items = []
    for p in paths:
        file_name = os.path.basename(p)
        track_id = int(os.path.splitext(file_name)[0])
        try:
            genre = tracks.loc[track_id, ('track', 'genre_top')]
            items.append((p, genre))
        except Exception:
            continue  # skip if missing or error

    genres = [genre for (_, genre) in items]
    items_train, items_test = train_test_split(items, test_size=0.2, stratify=genres)

    def parallel_process(items, desc):
        output = []
        with ProcessPoolExecutor(max_workers=6) as executor:
            futures = {executor.submit(process_file, item): item for item in items}
            for future in tqdm(as_completed(futures), total=len(items), desc=desc):
                try:
                    result = future.result()
                    if result:
                        output.extend(result)
                except Exception as e:
                    print(f"Subprocess crashed on item {futures[future]}: {type(e).__name__}: {e}")
        return output

    features_list_train = parallel_process(items_train, desc='Extracting features for train')
    features_list_test = parallel_process(items_test, desc='Extracting features for test')

    df_train = pd.DataFrame(features_list_train)
    df_test = pd.DataFrame(features_list_test)

    print('df train shape :', df_train.shape)
    print('df test shape :', df_test.shape)

    df_train.columns = col_names
    df_test.columns = col_names

    df_train = df_train.set_index('track_id')
    df_test = df_test.set_index('track_id')

    df_train.to_csv('fma_train.csv')
    df_test.to_csv('fma_test.csv')

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()