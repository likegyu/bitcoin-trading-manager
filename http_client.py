# =============================================
# 보안 강화 HTTP 세션 (공유 모듈)
# =============================================
# trust_env=False : HTTP_PROXY / HTTPS_PROXY 환경변수를 무시하여
#                   주입된 프록시를 통한 API 키 탈취(MITM)를 방지합니다.
import requests

_session = requests.Session()
_session.trust_env = False   # 프록시 환경변수 무시 — MITM 방지
