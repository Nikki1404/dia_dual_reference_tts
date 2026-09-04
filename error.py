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

env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
curl -v https://registry-1.docker.io/v2/
