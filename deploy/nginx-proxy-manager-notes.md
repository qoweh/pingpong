# Nginx Proxy Manager Notes

웹 서비스 앞에 Nginx Proxy Manager를 둘 때 필요한 최소 설정이다.

Proxy host target:

```text
Scheme: http
Forward Hostname/IP: server internal IP
Forward Port: 8079
```

## 권장 설정

- Enable Websockets. 실시간 시뮬레이션 상태는 `/api/live`를 사용한다.
- 개발 중에는 cache를 꺼도 된다.
- 배포가 안정되면 hashed asset에 긴 cache header를 줄 수 있다.
- `.wasm`, `.mjb`, `.js`, `.css` 파일 압축을 켠다.
- SSL can be managed by Nginx Proxy Manager.
