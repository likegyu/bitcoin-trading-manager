#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/crypto_analyzer}"
SERVICE_NAME="${SERVICE_NAME:-crypto-analyzer}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v sudo >/dev/null 2>&1; then
  echo "sudo가 필요합니다."
  exit 1
fi

if [ ! -f "${APP_DIR}/requirements.txt" ]; then
  echo "requirements.txt를 찾지 못했습니다: ${APP_DIR}"
  echo "먼저 프로젝트를 ${APP_DIR}에 업로드하거나 APP_DIR 환경변수를 맞춰주세요."
  exit 1
fi

echo "==> apt 패키지 설치"
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip

echo "==> 가상환경 준비"
if [ ! -d "${APP_DIR}/.venv" ]; then
  "${PYTHON_BIN}" -m venv "${APP_DIR}/.venv"
fi

"${APP_DIR}/.venv/bin/pip" install --upgrade pip
"${APP_DIR}/.venv/bin/pip" install -r "${APP_DIR}/requirements.txt"

if [ ! -f "${APP_DIR}/.env" ]; then
  echo "==> .env 파일이 없습니다. ${APP_DIR}/.env 를 먼저 만들어 주세요."
  exit 1
fi

echo "==> systemd 서비스 파일 생성"
tmp_service="$(mktemp)"
sed \
  -e "s|__APP_DIR__|${APP_DIR}|g" \
  -e "s|__SERVICE_NAME__|${SERVICE_NAME}|g" \
  "${APP_DIR}/deploy/lightsail/crypto-analyzer.service.template" > "${tmp_service}"

sudo cp "${tmp_service}" "/etc/systemd/system/${SERVICE_NAME}.service"
rm -f "${tmp_service}"

echo "==> 서비스 등록 및 시작"
sudo systemctl daemon-reload
sudo systemctl enable --now "${SERVICE_NAME}.service"
sudo systemctl restart "${SERVICE_NAME}.service"

echo
echo "완료:"
echo "  서비스 상태: sudo systemctl status ${SERVICE_NAME}"
echo "  로그 보기:   sudo journalctl -u ${SERVICE_NAME} -f"
