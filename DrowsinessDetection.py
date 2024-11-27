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
        writer.writerow(["Timestamp", "ErrorCount", "RRCount", "SDNN", "LF", "HF", "LF/HF", "SD1", "SD2"])

# 처리된 타임스탬프 저장
processed_timestamps = set()

def fetch_rr_intervals():
    """
    Firebase에서 RR 간격 데이터를 가져옵니다.
    """
    ref = db.reference('HeartRateData')
    all_data = ref.get()

    if not all_data:
        print("No data found in Firebase.") 
        return []

    rr_data_list = []
    for key, value in all_data.items():
        if key not in processed_timestamps:  # 새 데이터만 처리
            if "rrIntervals" in value:
                rr_intervals = [item["rrInterval"] for item in value["rrIntervals"]]
                rr_data_list.append((key, rr_intervals, value.get("errorCount", 0)))
            else:
                rr_data_list.append((key, [], value.get("errorCount", 0)))

    return rr_data_list

def calculate_time_domain_hrv(rr_intervals):
    if len(rr_intervals) < 2:
        return {"SDNN": None}
    
    sdnn = np.std(rr_intervals, ddof=1)
    return {"SDNN": sdnn}

def calculate_frequency_domain_hrv(rr_intervals):
    if len(rr_intervals) < 30:  # 데이터 부족
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

def write_to_csv(timestamp, error_count, rr_count, hrv_data):
    with open(CSV_FILE_PATH, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            timestamp,
            error_count,
            rr_count,  # 실제 RR 간격 개수
            hrv_data.get("SDNN"),
            hrv_data.get("LF"),
            hrv_data.get("HF"),
            hrv_data.get("LF/HF"),
            hrv_data.get("SD1"),
            hrv_data.get("SD2")
        ])

def process_rr_data():
    """
    Firebase 데이터 처리:
    1. RR 간격 데이터를 가져옴
    2. HRV 계산
    3. CSV 파일에 결과 저장
    """
    rr_data_list = fetch_rr_intervals()
    last_hrv_data = {"SDNN": None, "LF": None, "HF": None, "LF/HF": None, "SD1": None, "SD2": None}

    for timestamp, rr_intervals, error_count in rr_data_list:
        print(f"Processing data for timestamp: {timestamp}")
        print(f"Error Count: {error_count}")
        print(f"RR Intervals: {len(rr_intervals)}")

        # ErrorCount와 RR 리스트 길이 검증
        if len(rr_intervals) + error_count != 120:
            print(f"Warning: Data inconsistency detected for timestamp {timestamp}")
            continue

        if len(rr_intervals) < 100:  # 데이터 부족
            print(f"Insufficient data for HRV calculation at {timestamp}. Skipping.")
            write_to_csv(timestamp, error_count, len(rr_intervals), last_hrv_data)
            continue

        # HRV 계산
        time_domain_hrv = calculate_time_domain_hrv(rr_intervals)
        frequency_domain_hrv = calculate_frequency_domain_hrv(rr_intervals)
        nonlinear_domain_hrv = calculate_nonlinear_domain_hrv(rr_intervals)

        hrv_data = {**time_domain_hrv, **frequency_domain_hrv, **nonlinear_domain_hrv}
        write_to_csv(timestamp, error_count, len(rr_intervals), hrv_data)

        last_hrv_data = hrv_data
        processed_timestamps.add(timestamp)  # 처리된 타임스탬프 기록

        print(f"HRV Calculated for {timestamp}: {hrv_data}")
        print("-" * 50)

if __name__ == "__main__":
    while True:
        process_rr_data()
        print("Waiting for the next interval...")
        time.sleep(120)
