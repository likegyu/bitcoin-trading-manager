#!/usr/bin/env bash
# BTC 단타 대시보드 — Lightsail 초기 설치 스크립트
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/crypto_analyzer}"
SERVICE_NAME="btc-dashboard"
DASH_DIR="${APP_DIR}/btc-dashboard"

echo "==> Node.js 설치 확인"
if ! command -v node >/dev/null 2>&1; then
  echo "Node.js 설치 중..."
  curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
  sudo apt-get install -y nodejs
fi
echo "Node.js $(node -v) / npm $(npm -v)"

echo "==> npm 패키지 설치"
cd "${DASH_DIR}"
npm install --production

echo "==> .env 확인"
if [ ! -f "${DASH_DIR}/.env" ]; then
  echo "[ERROR] ${DASH_DIR}/.env 파일이 없습니다."
  echo "        cp ${DASH_DIR}/.env.example ${DASH_DIR}/.env 후 API 키를 입력하세요."
  exit 1
fi

echo "==> systemd 서비스 등록"
tmp="$(mktemp)"
sed -e "s|__APP_DIR__|${APP_DIR}|g" \
    "${DASH_DIR}/deploy/btc-dashboard.service.template" > "${tmp}"
sudo mv "${tmp}" "/etc/systemd/system/${SERVICE_NAME}.service"
sudo systemctl daemon-reload
sudo systemctl enable "${SERVICE_NAME}"
sudo systemctl restart "${SERVICE_NAME}"

echo "==> 완료!"
echo "   대시보드: http://$(curl -s ifconfig.me):3000"
echo "   로그:     journalctl -u ${SERVICE_NAME} -f"
