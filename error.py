(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/dia_dual_reference_tts# docker build -t dia-final .
[+] Building 30.1s (2/2) FINISHED                                                                                                                          docker:default
 => [internal] load build definition from Dockerfile                                                                                                                 0.0s
 => => transferring dockerfile: 1.31kB                                                                                                                               0.0s
 => ERROR [internal] load metadata for docker.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04                                                                       30.0s
------
 > [internal] load metadata for docker.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04:
------
Dockerfile:1
--------------------
   1 | >>> FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
   2 |
   3 |     ENV http_proxy="http://163.116.128.80:8080"
--------------------
ERROR: failed to build: failed to solve: DeadlineExceeded: nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04: failed to resolve source metadata for docker.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04: failed to do request: Head "https://registry-1.docker.io/v2/nvidia/cuda/manifests/12.4.1-cudnn-runtime-ubuntu22.04": proxyconnect tcp: dial tcp 163.116.128.80:8080: i/o timeout

(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/dia_dual_reference_tts# curl -v -x http://163.116.128.80:8080 https://registry-1.docker.io/v2/
*   Trying 163.116.128.80:8080...
* TCP_NODELAY set
* connect to 163.116.128.80 port 8080 failed: Connection timed out
* Failed to connect to 163.116.128.80 port 8080: Connection timed out
* Closing connection 0
curl: (28) Failed to connect to 163.116.128.80 port 8080: Connection timed out

(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/dia_dual_reference_tts# env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
curl -v https://registry-1.docker.io/v2/
*   Trying 54.204.25.166:443...
* TCP_NODELAY set
* Connected to registry-1.docker.io (54.204.25.166) port 443 (#0)
* ALPN, offering http/1.1
* successfully set certificate verify locations:
*   CAfile: /root/anaconda3/ssl/cacert.pem
  CApath: none
* TLSv1.3 (OUT), TLS handshake, Client hello (1):
* TLSv1.3 (IN), TLS handshake, Server hello (2):
* TLSv1.3 (IN), TLS handshake, Encrypted Extensions (8):
* TLSv1.3 (IN), TLS handshake, Certificate (11):
* TLSv1.3 (IN), TLS handshake, CERT verify (15):
* TLSv1.3 (IN), TLS handshake, Finished (20):
* TLSv1.3 (OUT), TLS change cipher, Change cipher spec (1):
* TLSv1.3 (OUT), TLS handshake, Finished (20):
* SSL connection using TLSv1.3 / TLS_AES_128_GCM_SHA256
* ALPN, server accepted to use http/1.1
* Server certificate:
*  subject: CN=*.docker.com
*  start date: Aug 25 00:00:00 2026 GMT
*  expire date: Mar 10 23:59:59 2027 GMT
*  subjectAltName: host "registry-1.docker.io" matched cert's "*.docker.io"
*  issuer: C=US; O=Amazon; CN=Amazon RSA 2048 M01
*  SSL certificate verify ok.
> GET /v2/ HTTP/1.1
> Host: registry-1.docker.io
> User-Agent: curl/7.68.0
> Accept: */*
>
* TLSv1.3 (IN), TLS handshake, Newsession Ticket (4):
* Mark bundle as not supporting multiuse
< HTTP/1.1 401 Unauthorized
< Date: Fri, 04 Sep 2026 08:59:14 GMT
< Content-Type: application/json
< Content-Length: 87
< Connection: keep-alive
< docker-distribution-api-version: registry/2.0
< www-authenticate: Bearer realm="https://auth.docker.io/token",service="registry.docker.io"
< strict-transport-security: max-age=31536000
<
{"errors":[{"code":"UNAUTHORIZED","message":"authentication required","detail":null}]}
* Connection #0 to host registry-1.docker.io left intact


(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/dia_dual_reference_tts# docker info | grep -i proxy
 HTTP Proxy: http://163.116.128.80:8080
 HTTPS Proxy: http://163.116.128.80:8080
 No Proxy: localhost,127.0.0.1,169.254.169.254,metadata.google.internal
  EnableUserlandProxy: true
  UserlandProxyPath: /usr/bin/docker-proxy
(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/dia_dual_reference_tts# systemctl show docker --property=Environment
Environment=HTTP_PROXY=http://163.116.128.80:8080 HTTPS_PROXY=http://163.116.128.80:8080 NO_PROXY=localhost,127.0.0.1,169.254.169.254,metadata.google.internal
