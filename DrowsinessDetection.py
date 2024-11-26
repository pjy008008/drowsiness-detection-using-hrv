import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import numpy as np
import time
import csv
import os

# Firebase 인증 및 초기화
cred = credentials.Certificate("firebase\hrvdataset-firebase-adminsdk-oof96-146efebb50.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://hrvdataset-default-rtdb.firebaseio.com/'
})

# CSV 파일 초기화
CSV_FILE_PATH = "hrv_results.csv"

if not os.path.exists(CSV_FILE_PATH):
    with open(CSV_FILE_PATH, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "ErrorCount", "SDNN", "LF", "HF", "LF/HF", "SD1", "SD2"])

def fetch_rr_intervals():
    ref = db.reference('HeartRateData')
    all_data = ref.get()

    if not all_data:
        print("No data found in Firebase.")
        return []

    rr_data_list = []
    for key, value in all_data.items():
        if "rrIntervals" in value:
            rr_intervals = [item["rrInterval"] for item in value["rrIntervals"]]
            rr_data_list.append((key, rr_intervals, value.get("errorCount", 0)))
        else:
            # Add an empty RR interval list for entries without rrIntervals
            rr_data_list.append((key, [], value.get("errorCount", 0)))
    
    return rr_data_list

def calculate_time_domain_hrv(rr_intervals):
    if len(rr_intervals) < 2:
        return {"SDNN": None}
    
    sdnn = np.std(rr_intervals, ddof=1)
    return {"SDNN": sdnn}

def calculate_frequency_domain_hrv(rr_intervals):
    if len(rr_intervals) < 2:
        return {"LF": None, "HF": None, "LF/HF": None}
    
    from scipy.signal import welch
    freq, power = welch(rr_intervals, fs=4.0, nperseg=len(rr_intervals))

    lf_band = (0.04, 0.15)
    hf_band = (0.15, 0.4)

    lf_power = np.trapz(power[(freq >= lf_band[0]) & (freq < lf_band[1])], freq[(freq >= lf_band[0]) & (freq < lf_band[1])])
    hf_power = np.trapz(power[(freq >= hf_band[0]) & (freq < hf_band[1])], freq[(freq >= hf_band[0]) & (freq < hf_band[1])])
    lf_hf_ratio = lf_power / hf_power if hf_power > 0 else None

    return {"LF": lf_power, "HF": hf_power, "LF/HF": lf_hf_ratio}

def calculate_nonlinear_domain_hrv(rr_intervals):
    if len(rr_intervals) < 2:
        return {"SD1": None, "SD2": None}
    
    diff_rr = np.diff(rr_intervals)
    sd1 = np.std(diff_rr) / np.sqrt(2)
    sd2 = np.sqrt(2 * np.std(rr_intervals, ddof=1)**2 - sd1**2)
    return {"SD1": sd1, "SD2": sd2}

def write_to_csv(timestamp, error_count, hrv_data):
    with open(CSV_FILE_PATH, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            timestamp,
            error_count,
            hrv_data.get("SDNN"),
            hrv_data.get("LF"),
            hrv_data.get("HF"),
            hrv_data.get("LF/HF"),
            hrv_data.get("SD1"),
            hrv_data.get("SD2")
        ])

def process_rr_data():
    rr_data_list = fetch_rr_intervals()
    last_hrv_data = {"SDNN": None, "LF": None, "HF": None, "LF/HF": None, "SD1": None, "SD2": None}

    for timestamp, rr_intervals, error_count in rr_data_list:
        print(f"Processing data for timestamp: {timestamp}")
        print(f"Error Count: {error_count}")
        print(f"RR Intervals: {rr_intervals}")

        if error_count >= 10:
            print("Error count too high. Skipping HRV calculation.")
            write_to_csv(timestamp, error_count, last_hrv_data)
            continue

        time_domain_hrv = calculate_time_domain_hrv(rr_intervals)
        frequency_domain_hrv = calculate_frequency_domain_hrv(rr_intervals)
        nonlinear_domain_hrv = calculate_nonlinear_domain_hrv(rr_intervals)

        hrv_data = {**time_domain_hrv, **frequency_domain_hrv, **nonlinear_domain_hrv}
        write_to_csv(timestamp, error_count, hrv_data)

        # Update the last valid HRV data
        last_hrv_data = hrv_data

        print(f"Time Domain HRV: {time_domain_hrv}")
        print(f"Frequency Domain HRV: {frequency_domain_hrv}")
        print(f"Nonlinear Domain HRV: {nonlinear_domain_hrv}")
        print("-" * 50)

if __name__ == "__main__":
    while True:
        process_rr_data()
        print("Waiting for the next interval...")
        time.sleep(120)  # 2분 간격
