# Nginx Proxy Manager Notes

Proxy host target:

```text
Scheme: http
Forward Hostname/IP: home server internal IP
Forward Port: 8079
```

Recommended settings:

- Websockets are not required for the MVP.
- Cache can remain off while assets are changing.
- SSL can be managed by Nginx Proxy Manager.
