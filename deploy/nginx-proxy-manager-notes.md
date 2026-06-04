# Nginx Proxy Manager Notes

Proxy host target:

```text
Scheme: http
Forward Hostname/IP: home server internal IP
Forward Port: 8079
```

Recommended settings:

- Enable Websockets. The live RL state stream uses `/api/live`.
- Cache can remain off while assets are changing.
- SSL can be managed by Nginx Proxy Manager.
