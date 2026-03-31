# Lightsail 배포 가이드

이 프로젝트는 AWS Lightsail에서 `Ubuntu + systemd + uvicorn` 조합으로 가장 단순하게 배포할 수 있습니다.

## 추천 스펙

- 리전: `Seoul`
- 인스턴스: `Linux/Unix`
- 블루프린트: `OS Only > Ubuntu 22.04 LTS`
- 플랜: `1 GB RAM 이상`
- 네트워크: `Static IP` 연결

## 1. Lightsail 인스턴스 준비

1. Ubuntu 인스턴스를 생성합니다.
2. `Networking` 탭에서 `Static IP`를 생성해 인스턴스에 연결합니다.
3. 방화벽에서 `8000/TCP`를 엽니다.
4. Binance API 화이트리스트에 이 Static IP를 등록합니다.

## 2. 서버 접속 후 프로젝트 업로드

```bash
ssh ubuntu@YOUR_STATIC_IP
sudo mkdir -p /opt
cd /opt
sudo git clone <YOUR_REPO_URL> crypto_analyzer
sudo chown -R $USER:$USER /opt/crypto_analyzer
cd /opt/crypto_analyzer
```

git 대신 압축 업로드를 써도 됩니다. 중요한 것은 최종 경로가 `/opt/crypto_analyzer`이거나, 아래 `APP_DIR` 값을 맞추는 것입니다.

## 3. 환경 변수 파일 준비

`/opt/crypto_analyzer/.env` 파일을 만들고 필요한 키를 넣습니다.

```env
CLAUDE_API_KEY=...
BINANCE_API_KEY=...
BINANCE_SECRET_KEY=...
FRED_API_KEY=...
DEFAULT_SYMBOL=BTCUSDC
```

## 4. 앱 설치 및 서비스 등록

```bash
cd /opt/crypto_analyzer
chmod +x deploy/lightsail/bootstrap_ubuntu.sh
APP_DIR=/opt/crypto_analyzer ./deploy/lightsail/bootstrap_ubuntu.sh
```

기본 서비스 이름은 `crypto-analyzer`입니다. 바꾸고 싶으면:

```bash
APP_DIR=/opt/crypto_analyzer SERVICE_NAME=crypto-analyzer ./deploy/lightsail/bootstrap_ubuntu.sh
```

## 5. 상태 확인

```bash
sudo systemctl status crypto-analyzer
sudo journalctl -u crypto-analyzer -f
```

브라우저에서 아래 주소로 접속합니다.

```text
http://YOUR_STATIC_IP:8000
```

## 운영 메모

- 이 서비스는 `uvicorn --reload` 없이 실행됩니다.
- 코드 업데이트 후에는 아래처럼 반영하면 됩니다.

```bash
cd /opt/crypto_analyzer
git pull
.venv/bin/pip install -r requirements.txt
sudo systemctl restart crypto-analyzer
```

- HTTPS가 필요해지면 나중에 `Caddy`나 `Nginx`를 앞단에 붙이면 됩니다.
