import os
import random
import numpy as np
import scipy.io as io
import mne

# Global Constants
ROOT_DIR = './data'
SAVE_DIR_NORMAL = './data/dataset_chb/MAT/normal/3s769'
SAVE_DIR_ABNORMAL = './data/dataset_chb/MAT/abnormal/3s769'
TRAIN_FILE = './data/all_TXT_3s/train.txt'
TEST_FILE = './data/all_TXT_3s/test.txt'
SEGMENT_LENGTH = 3  # Segment length in seconds
SELECTION = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3',
    'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2', 'FZ-CZ', 'CZ-PZ'
]

# Ensure directories exist
os.makedirs(SAVE_DIR_NORMAL, exist_ok=True)
os.makedirs(SAVE_DIR_ABNORMAL, exist_ok=True)
os.makedirs(os.path.dirname(TRAIN_FILE), exist_ok=True)

def process_files(subject_dir, save_dir, abnormal=False):
    """
    Process EDF files from a subject directory.
    Crops and saves the EEG data to FIF format.
    """
    total = 0
    subject_path = os.path.join(ROOT_DIR, subject_dir)
    save_subject_dir = os.path.join(save_dir, subject_dir)
    os.makedirs(save_subject_dir, exist_ok=True)
    
    # Collect files
    edf_files = []
    annotations = {}
    for root, _, files in os.walk(subject_path):
        for file in files:
            if abnormal and file.endswith(".seizures"):
                edf_files.append(file[:-9])
            elif file.endswith(".edf"):
                edf_files.append(file)
            elif "summary.txt" in file:
                with open(os.path.join(root, file), "r") as f:
                    for line in f.readlines():
                        parts = line.split(":")
                        if len(parts) == 2:  # Skip malformed lines
                            annotations[parts[0].strip()] = parts[1].strip()
    
    # Process each EDF file
    for file in edf_files:
        edf_path = os.path.join(subject_path, file)
        try:
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            raw.set_meas_date(None)  # Remove metadata for anonymization
            
            if abnormal and file in annotations:
                seizure_times = annotations[file].split('-')
                start, end = int(seizure_times[0]), int(seizure_times[1])
                raw.crop(tmin=start, tmax=end)
            
            save_path = os.path.join(save_subject_dir, file.replace(".edf", f"_{total}_raw.fif"))
            raw.save(save_path, overwrite=True)
            total += 1
        except Exception as e:
            print(f"Error processing file {edf_path}: {e}")


def generate_mat_files(input_dir, save_dir, segment_length):
    """
    Generate EEG data in .mat format segmented into fixed intervals.
    """
    os.makedirs(save_dir, exist_ok=True)
    total = 0
    for root, _, files in os.walk(input_dir):
        for file in files:
            raw_path = os.path.join(root, file)
            try:
                raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
                available_channels = raw.ch_names
                missing_channels = [ch for ch in SELECTION if ch not in available_channels]

                if missing_channels:
                    print(f"Skipping {raw_path}: Missing channels {missing_channels}")
                    continue

                raw.pick_channels(SELECTION)

                data = raw.get_data()
                sfreq = raw.info['sfreq']
                segment_samples = int(segment_length * sfreq)
                length = data.shape[1]

                for start in range(0, length, segment_samples):
                    end = start + segment_samples
                    if end > length:
                        break
                    segment_data = data[:, start:end]
                    save_path = os.path.join(save_dir, f"{file[:-4]}_{total}.mat")
                    io.savemat(save_path, {'data': segment_data})
                    total += 1
            except Exception as e:
                print(f"Error processing file {raw_path}: {e}")


def save_paths_to_txt(normal_dir, abnormal_dir, train_file, test_file):
    """
    Save the paths of processed .mat files into train and test text files.
    """
    normal_paths = [
        os.path.join(root, file)
        for root, _, files in os.walk(normal_dir) for file in files
    ]
    abnormal_paths = [
        os.path.join(root, file)
        for root, _, files in os.walk(abnormal_dir) for file in files
    ]
    
    random.shuffle(normal_paths)
    random.shuffle(abnormal_paths)
    
    with open(train_file, 'w') as train_f:
        train_f.write("\n".join(normal_paths))
    with open(test_file, 'w') as test_f:
        test_f.write("\n".join(abnormal_paths))

if __name__ == "__main__":
    # Subject directories
    dirs_normal = ['chb01', 'chb06', 'chb09', 'chb11', 'chb20', 'chb21', 'chb23']
    dirs_abnormal = ['chb01', 'chb02', 'chb03', 'chb04', 'chb05', 
                     'chb06', 'chb07', 'chb08', 'chb09', 'chb10', 
                     'chb11', 'chb12', 'chb13', 'chb14', 'chb15', 
                     'chb16', 'chb17', 'chb18', 'chb19', 'chb20', 
                     'chb21', 'chb22', 'chb23']
    
    # Process and save normal EEG
    for subject in dirs_normal:
        process_files(subject, SAVE_DIR_NORMAL, abnormal=False)
    
    # Process and save abnormal EEG
    for subject in dirs_abnormal:
        process_files(subject, SAVE_DIR_ABNORMAL, abnormal=True)
    
    # Generate .mat files for normal and abnormal EEG
    generate_mat_files(SAVE_DIR_NORMAL, SAVE_DIR_NORMAL, SEGMENT_LENGTH)
    generate_mat_files(SAVE_DIR_ABNORMAL, SAVE_DIR_ABNORMAL, SEGMENT_LENGTH)
    
    # Save paths for train and test datasets
    save_paths_to_txt(SAVE_DIR_NORMAL, SAVE_DIR_ABNORMAL, TRAIN_FILE, TEST_FILE)

