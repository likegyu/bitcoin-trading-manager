/**
 * BTC 단타 대시보드 — Lightsail 서버
 * - Binance REST API 프록시 (서버 IP로 호출 → 한국 IP 차단 우회)
 * - Binance WebSocket 중계 (서버가 fstream에 연결 후 브라우저에 전달)
 * - public/ 정적 파일 서빙
 */

const express  = require('express');
const http     = require('http');
const https    = require('https');
const crypto   = require('crypto');
const WebSocket = require('ws');
const path     = require('path');
require('dotenv').config();

const API_KEY   = process.env.BINANCE_API_KEY;
const SECRET    = process.env.BINANCE_SECRET_KEY;
const PORT      = process.env.PORT || 3000;

if (!API_KEY || !SECRET) {
  console.error('[ERROR] .env에 BINANCE_API_KEY / BINANCE_SECRET_KEY가 없습니다.');
  process.exit(1);
}

/* ── 서버 시간 오프셋 ── */
let serverTimeOffset = 0;
async function syncTime() {
  return new Promise(resolve => {
    https.get('https://fapi.binance.com/fapi/v1/time', res => {
      let body = '';
      res.on('data', d => body += d);
      res.on('end', () => {
        try {
          const { serverTime } = JSON.parse(body);
          serverTimeOffset = serverTime - Date.now();
          console.log(`[시간 동기화] 오프셋: ${serverTimeOffset}ms`);
        } catch(_) {}
        resolve();
      });
    }).on('error', () => resolve());
  });
}

/* ── HMAC-SHA256 서명 ── */
function sign(queryString) {
  return crypto.createHmac('sha256', SECRET).update(queryString).digest('hex');
}

/* ── Binance REST 프록시 유틸 ── */
// 서명이 필요한 private 엔드포인트 목록
const PRIVATE_PATHS = [
  '/fapi/v2/account',
  '/fapi/v2/positionRisk',
  '/fapi/v1/openOrders',
  '/fapi/v1/order',
];

function isPrivate(urlPath) {
  return PRIVATE_PATHS.some(p => urlPath.startsWith(p));
}

function binanceFetch(urlPath, queryParams) {
  return new Promise((resolve, reject) => {
    let qs = new URLSearchParams(queryParams).toString();
    if (isPrivate(urlPath)) {
      const ts = Date.now() + serverTimeOffset;
      const base = qs ? `${qs}&timestamp=${ts}&recvWindow=5000`
                      : `timestamp=${ts}&recvWindow=5000`;
      const sig = sign(base);
      qs = `${base}&signature=${sig}`;
    }
    const url = `https://fapi.binance.com${urlPath}${qs ? '?' + qs : ''}`;
    const opts = { headers: { 'X-MBX-APIKEY': API_KEY } };
    https.get(url, opts, res => {
      let body = '';
      res.on('data', d => body += d);
      res.on('end', () => resolve({ status: res.statusCode, body }));
    }).on('error', reject);
  });
}

/* ── Express 앱 ── */
const app = express();

// 루트 접속 시 대시보드로 바로 이동
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'btc-dashboard.html'));
});

// 정적 파일 (public/btc-dashboard.html 등)
app.use(express.static(path.join(__dirname, 'public')));

// Binance REST 프록시
// 브라우저: fetch('/proxy/fapi/v2/account') → 서버가 서명해서 Binance에 요청
app.get('/proxy/*', async (req, res) => {
  const urlPath = req.path.replace('/proxy', ''); // /fapi/v2/account 등
  const queryParams = req.query;                  // URL 파라미터 그대로 전달
  try {
    const { status, body } = await binanceFetch(urlPath, queryParams);
    res.status(status).set('Content-Type', 'application/json').send(body);
  } catch(e) {
    console.error('[proxy error]', urlPath, e.message);
    res.status(502).json({ error: 'proxy_error', message: e.message });
  }
});

/* ── HTTP 서버 + WebSocket 서버 ── */
const server = http.createServer(app);

// WebSocket 중계: 브라우저 → 서버 → Binance fstream
const wss = new WebSocket.Server({ server, path: '/ws' });

wss.on('connection', (clientWs, req) => {
  const qs      = req.url.split('?')[1] || '';
  const streams = new URLSearchParams(qs).get('streams');
  if (!streams) { clientWs.close(); return; }

  const binanceUrl = `wss://fstream.binance.com/stream?streams=${streams}`;
  console.log(`[WS] 새 연결 → ${binanceUrl}`);

  const binanceWs = new WebSocket(binanceUrl);

  binanceWs.on('open',  ()    => console.log('[WS] Binance 연결됨'));
  binanceWs.on('message', data => {
    if (clientWs.readyState === WebSocket.OPEN) clientWs.send(data.toString());
  });
  binanceWs.on('error', e   => console.error('[WS Binance error]', e.message));
  binanceWs.on('close', ()  => {
    console.log('[WS] Binance 연결 종료');
    if (clientWs.readyState === WebSocket.OPEN) clientWs.close();
  });

  clientWs.on('close', () => {
    console.log('[WS] 브라우저 연결 종료');
    if (binanceWs.readyState === WebSocket.OPEN) binanceWs.close();
  });
  clientWs.on('error', e => console.error('[WS client error]', e.message));
});

/* ── 시작 ── */
syncTime().then(() => {
  server.listen(PORT, '0.0.0.0', () => {
    console.log(`✅ 서버 시작: http://0.0.0.0:${PORT}`);
    console.log(`   대시보드: http://[Lightsail_IP]:${PORT}`);
  });
});

// 매 10분마다 서버 시간 재동기화
setInterval(syncTime, 10 * 60 * 1000);
