import firebase_admin
from firebase_admin import credentials, auth
import datetime
import jwt  # PyJWT 필요

# 기존 Firebase 앱 삭제 후 재초기화
if firebase_admin._apps:
    firebase_admin.delete_app(firebase_admin.get_app())

cred = credentials.Certificate("hrv-data-a12d2-firebase-adminsdk-fbsvc-d6a2051332.json")
firebase_admin.initialize_app(cred, {"databaseURL": "https://hrv-data-a12d2-default-rtdb.firebaseio.com/"})

# 서비스 계정 이메일 가져오기
service_account_email = cred.service_account_email

# Firebase Custom Token 생성 (1시간 유효)
custom_token = auth.create_custom_token(service_account_email)

# JWT 토큰 디코딩
decoded_token = jwt.decode(custom_token, options={"verify_signature": False})

# 발급 시간(iat) 및 만료 시간(exp) 확인
iat_timestamp = decoded_token.get("iat")
exp_timestamp = decoded_token.get("exp")

iat_datetime = datetime.datetime.fromtimestamp(iat_timestamp)
exp_datetime = datetime.datetime.fromtimestamp(exp_timestamp)

print(f"🔹 토큰 발급 시간 (iat): {iat_timestamp} ({iat_datetime})")
print(f"🔹 토큰 만료 시간 (exp): {exp_timestamp} ({exp_datetime})")
