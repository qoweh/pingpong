# Nginx Proxy Manager Notes

웹 서비스 앞에 Nginx Proxy Manager를 둘 때 필요한 최소 설정이다.

Proxy host target:

```text
Scheme: http
Forward Hostname/IP: server internal IP
Forward Port: 8079
```

## 권장 설정

- Websockets를 켠다. 실시간 시뮬레이션 stream은 `WS /api/live`를 사용한다.
- `/api/*`는 cache하지 않는다. 모델 목록, health, WebSocket 전환 상태가 즉시 반영되어야 한다.
- `/assets/*`와 Vite hashed asset은 긴 cache header를 줄 수 있다.
- `.wasm`, `.mjb`, `.js`, `.css`는 gzip 또는 Brotli 압축을 켠다.
- `pingpong_scene.mjb`와 MuJoCo WASM은 첫 로딩 용량이 크므로 proxy upload/download limit을 너무 낮게 두지 않는다.
- SSL 인증서는 Nginx Proxy Manager에서 관리하면 된다.

## 빠른 점검

```sh
curl http://<proxy-host>/api/health
```

브라우저 개발자 도구에서 `/api/live`가 `101 Switching Protocols`로 연결되면 WebSocket proxy는 정상이다.
