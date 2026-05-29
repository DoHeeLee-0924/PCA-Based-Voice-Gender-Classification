import os
import numpy as np
import pandas as pd
import librosa


# ------------------------------------------------------------
# 1. Path setting
# ------------------------------------------------------------

PROJECT_DIR = r"C:\Users\ldohe\Documents\다변량분석\project"

FEMALE_DIR = os.path.join(PROJECT_DIR, "data", "female")
MALE_DIR = os.path.join(PROJECT_DIR, "data", "male")
TEST_DIR = os.path.join(PROJECT_DIR, "data", "test")

OUTPUT_TRAIN_CSV = os.path.join(PROJECT_DIR, "voice_features_all.csv")
OUTPUT_TEST_CSV = os.path.join(PROJECT_DIR, "voice_features_test.csv")


print("PROJECT_DIR:", PROJECT_DIR)
print("FEMALE_DIR:", FEMALE_DIR)
print("MALE_DIR:", MALE_DIR)
print("TEST_DIR:", TEST_DIR)

print("female exists:", os.path.exists(FEMALE_DIR))
print("male exists:", os.path.exists(MALE_DIR))
print("test exists:", os.path.exists(TEST_DIR))


# ------------------------------------------------------------
# 2. Feature extraction
# ------------------------------------------------------------

def extract_features(file_path, sr=16000):
    """
    Extract compact acoustic features from one audio file.
    
    Extracted features:
    - MFCC0 ~ MFCC12
    - SpectralCentroid
    - SpectralSpread
    - SpectralRolloffPoint
    - PitchMean
    - PitchStd
    """

    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=sr, mono=True)

        # If empty or silent, skip
        if len(y) == 0 or np.max(np.abs(y)) == 0:
            print(f"[SKIP] Silent or empty file: {file_path}")
            return None

        # Normalize
        y = librosa.util.normalize(y)

        features = {}

        # File name
        features["FileName"] = os.path.basename(file_path)

        # ----------------------------------------------------
        # 1) MFCC: 13 coefficients
        # ----------------------------------------------------
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=13,
            n_fft=1024,
            hop_length=512
        )

        mfcc_mean = np.mean(mfcc, axis=1)

        for i in range(13):
            features[f"MFCC{i}"] = mfcc_mean[i]

        # ----------------------------------------------------
        # 2) Spectral centroid
        # ----------------------------------------------------
        spectral_centroid = librosa.feature.spectral_centroid(
            y=y,
            sr=sr,
            n_fft=1024,
            hop_length=512
        )

        features["SpectralCentroid"] = np.mean(spectral_centroid)

        # ----------------------------------------------------
        # 3) Spectral spread
        # librosa에는 SpectralSpread라는 함수가 따로 없어서
        # spectral_bandwidth를 SpectralSpread로 사용
        # ----------------------------------------------------
        spectral_spread = librosa.feature.spectral_bandwidth(
            y=y,
            sr=sr,
            n_fft=1024,
            hop_length=512
        )

        features["SpectralSpread"] = np.mean(spectral_spread)

        # ----------------------------------------------------
        # 4) Spectral rolloff point
        # ----------------------------------------------------
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=y,
            sr=sr,
            n_fft=1024,
            hop_length=512,
            roll_percent=0.85
        )

        features["SpectralRolloffPoint"] = np.mean(spectral_rolloff)

        # ----------------------------------------------------
        # 5) Pitch features
        # ----------------------------------------------------
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
            frame_length=1024,
            hop_length=512
        )

        if f0 is not None and np.sum(~np.isnan(f0)) > 0:
            features["PitchMean"] = np.nanmean(f0)
            features["PitchStd"] = np.nanstd(f0)
        else:
            features["PitchMean"] = np.nan
            features["PitchStd"] = np.nan

        return features

    except Exception as e:
        print(f"[ERROR] {file_path}: {e}")
        return None


# ------------------------------------------------------------
# 3. Extract features from folder
# ------------------------------------------------------------

def extract_folder_features(folder_path, gender=None):
    """
    Extract features from all mp3 files in a folder.
    """

    rows = []

    files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(".mp3")
    ]

    files = sorted(files)

    print("\nFolder:", folder_path)
    print("Number of files:", len(files))

    for idx, file_path in enumerate(files, start=1):
        print(f"[{idx}/{len(files)}] Processing: {os.path.basename(file_path)}")

        features = extract_features(file_path)

        if features is not None:
            if gender is not None:
                features["gender"] = gender

            rows.append(features)

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# 4. Main
# ------------------------------------------------------------

def main():
    # Female and male training data
    female_df = extract_folder_features(FEMALE_DIR, gender="female")
    male_df = extract_folder_features(MALE_DIR, gender="male")

    train_df = pd.concat([female_df, male_df], ignore_index=True)

    # Column order
    feature_columns = (
        ["FileName"] +
        [f"MFCC{i}" for i in range(13)] +
        [
            "SpectralCentroid",
            "SpectralSpread",
            "SpectralRolloffPoint",
            "PitchMean",
            "PitchStd",
            "gender"
        ]
    )

    train_df = train_df[feature_columns]

    # Save training feature CSV
    train_df.to_csv(
        OUTPUT_TRAIN_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nSaved training feature CSV:")
    print(OUTPUT_TRAIN_CSV)
    print("Shape:", train_df.shape)
    print(train_df.head())

    # Test data without labels
    if os.path.exists(TEST_DIR):
        test_df = extract_folder_features(TEST_DIR, gender=None)

        test_feature_columns = (
            ["FileName"] +
            [f"MFCC{i}" for i in range(13)] +
            [
                "SpectralCentroid",
                "SpectralSpread",
                "SpectralRolloffPoint",
                "PitchMean",
                "PitchStd"
            ]
        )

        test_df = test_df[test_feature_columns]

        test_df.to_csv(
            OUTPUT_TEST_CSV,
            index=False,
            encoding="utf-8-sig"
        )

        print("\nSaved test feature CSV:")
        print(OUTPUT_TEST_CSV)
        print("Shape:", test_df.shape)
        print(test_df.head())


if __name__ == "__main__":
    main()