import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import numpy as np
import csv
import os
import time
from collections import deque

# Firebase 인증 및 초기화
cred = credentials.Certificate("firebase/hrvdataset-firebase-adminsdk-oof96-146efebb50.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://hrvdataset-default-rtdb.firebaseio.com/'
})

# CSV 파일 초기화
CSV_FILE_PATH = "hrv_results.csv"
TEMP_CSV_FILE_PATH = "temp_hrv_results.csv"

if not os.path.exists(CSV_FILE_PATH):
    with open(CSV_FILE_PATH, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["StartTimestamp", "EndTimestamp", "ErrorCount", "RRCount", "SDNN", "LF", "HF", "LF/HF", "SD1", "SD2"])

# Firebase에서 데이터 참조
ref = db.reference('HeartRateData')

# 슬라이딩 윈도우 데이터
rr_window = deque(maxlen=120)  # 최근 120초 데이터만 유지
last_processed_timestamp = None


def fetch_new_rr_intervals():
    """
    Firebase에서 새로운 RR 데이터를 가져옵니다.
    """
    global last_processed_timestamp
    all_data = ref.get()  # Firebase에서 모든 데이터를 가져옵니다.

    if not all_data:
        print("No data found in Firebase.")
        return []

    new_data = []
    for key, value in all_data.items():
        timestamp = value.get("timestamp")
        rr_interval = value.get("rrInterval", 0)
        is_error = value.get("isError", True)  # 기본값 True로 설정해 에러로 처리
        
        # 이미 처리된 데이터는 제외
        if last_processed_timestamp is None or timestamp > last_processed_timestamp:
            # 에러 데이터도 포함하되, rr_interval=0으로 처리
            if is_error or rr_interval <= 0:
                new_data.append((timestamp, 0))  # 에러 데이터는 rr_interval=0
            else:
                new_data.append((timestamp, rr_interval))


    # 새로운 데이터를 타임스탬프 기준으로 정렬
    new_data.sort(key=lambda x: x[0])

    return new_data


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

def write_to_csv(start_timestamp, end_timestamp, error_count, rr_count, hrv_data):
    # 기존 파일에서 데이터를 읽어와 임시 파일에 복사
    with open(TEMP_CSV_FILE_PATH, mode='w', newline='') as temp_file:
        writer = csv.writer(temp_file)
        
        # 기존 파일이 존재하면 내용을 복사
        if os.path.exists(CSV_FILE_PATH):
            with open(CSV_FILE_PATH, mode='r') as original_file:
                reader = csv.reader(original_file)
                for row in reader:
                    writer.writerow(row)
        
        # 새로운 데이터 추가
        writer.writerow([
            start_timestamp,
            end_timestamp,
            error_count,
            rr_count,
            hrv_data.get("SDNN"),
            hrv_data.get("LF"),
            hrv_data.get("HF"),
            hrv_data.get("LF/HF"),
            hrv_data.get("SD1"),
            hrv_data.get("SD2")
        ])
    
    # 기존 CSV 파일 삭제 후 임시 파일을 새 파일로 이동
    if os.path.exists(CSV_FILE_PATH):
        os.remove(CSV_FILE_PATH)  # 기존 파일 삭제
    os.rename(TEMP_CSV_FILE_PATH, CSV_FILE_PATH)  # 임시 파일을 새 파일로 교체

def process_hrv():
    """
    새로운 데이터를 가져와 슬라이딩 윈도우 방식으로 HRV를 계산합니다.
    """
    global last_processed_timestamp
    new_rr_data = fetch_new_rr_intervals()

    if not new_rr_data:
        print("No new data to process.")
        return

    for timestamp, rr_interval in new_rr_data:
        rr_window.append((timestamp, rr_interval))  # (timestamp, rr_interval) 형태로 저장

        if len(rr_window) == 120:  # 슬라이딩 윈도우가 꽉 찼을 때 HRV 계산
            start_timestamp = rr_window[0][0]  # 윈도우의 첫 번째 데이터의 timestamp
            end_timestamp = rr_window[-1][0]  # 윈도우의 마지막 데이터의 timestamp
            
            # 에러 데이터를 제외한 RR 간격만 추출
            rr_list = [item[1] for item in rr_window if item[1] > 0]  # RR 값이 0보다 큰 값만 추출
            error_count = 120 - len(rr_list)  # 총 데이터 수에서 유효한 데이터 수를 뺀 값

            # CSV 저장에 사용할 데이터 구성
            hrv_data = {"SDNN": None, "LF": None, "HF": None, "LF/HF": None, "SD1": None, "SD2": None}

            # 에러가 36회 이상이면 HRV 계산을 건너뜀
            if error_count >= 36:
                print(f"Too many errors ({error_count} errors). Skipping HRV calculation for window {start_timestamp} to {end_timestamp}.")
            else:
                # HRV 계산
                time_domain_hrv = calculate_time_domain_hrv(rr_list)
                frequency_domain_hrv = calculate_frequency_domain_hrv(rr_list)
                nonlinear_domain_hrv = calculate_nonlinear_domain_hrv(rr_list)

                # HRV 데이터 병합
                hrv_data = {**time_domain_hrv, **frequency_domain_hrv, **nonlinear_domain_hrv}
                print(f"HRV Calculated: {hrv_data}")

            # CSV 저장
            write_to_csv(start_timestamp, end_timestamp, error_count, len(rr_list), hrv_data)

            # 슬라이딩 윈도우를 1초씩 이동
            rr_window.popleft()

        # 마지막으로 처리한 타임스탬프 업데이트
        last_processed_timestamp = timestamp




if __name__ == "__main__":
    while True:
        process_hrv()
        print("Waiting for the next interval...")
        time.sleep(1)