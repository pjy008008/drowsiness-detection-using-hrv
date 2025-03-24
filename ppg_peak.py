#%%
import firebase_admin
from firebase_admin import credentials, db
import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt

# Firebase 인증 정보 불러오기 (JSON 키 파일 필요)
cred = credentials.Certificate("hrv-data-a12d2-firebase-adminsdk-fbsvc-d6a2051332.json")  # Firebase Admin SDK JSON 파일
firebase_admin.initialize_app(cred, {"databaseURL": "https://hrv-data-a12d2-default-rtdb.firebaseio.com/"})

# Firebase에서 데이터 가져오기
ref = db.reference("HeartRateData")
data = ref.get()

# 데이터 정리
ppg_values = []
timestamps = []

for key, value in data.items():
    if not value["isError"]:  # 오류 없는 데이터만 사용
        ppg_values.append(value["ppgGreen"])
        timestamps.append(pd.to_datetime(value["timestamp"]))

# 데이터프레임 생성
df = pd.DataFrame({"Timestamp": timestamps, "PPG": ppg_values})
df = df.sort_values("Timestamp")  # 시간 순 정렬

# PPG 신호 처리 (25Hz 가정)
fs = 25  # 샘플링 주파수
ppg_cleaned = nk.ppg_clean(df["PPG"], sampling_rate=fs)

# PPG 피크 검출
peaks, info = nk.ppg_peaks(ppg_cleaned, sampling_rate=fs)

# 시각화
plt.figure(figsize=(10, 5))
plt.plot(df["Timestamp"], ppg_cleaned, label="Cleaned PPG", color="blue")
plt.scatter(df["Timestamp"][peaks["PPG_Peaks"] == 1], ppg_cleaned[peaks["PPG_Peaks"] == 1], color="red", label="Peaks", zorder=3)
plt.legend()
plt.xlabel("Time")
plt.ylabel("PPG Signal")
plt.title("PPG Signal with Detected Peaks")
plt.xticks(rotation=45)
plt.grid()
plt.show()

# %%
