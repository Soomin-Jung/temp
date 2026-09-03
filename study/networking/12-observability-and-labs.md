# Observability, Labs, Failure Injection

업데이트: 2026-09-02

## 1. 목표

다음 가정을 직접 반증 가능한 실험으로 바꾼다.

- Service가 HTTP request마다 backend를 다시 고르는가?
- SSE event와 TCP read 경계가 같은가?
- EndpointSlice에서 Pod를 빼면 existing stream도 끊기는가?
- L7 router가 고른 Pod로 실제 packet이 가는가?
- Agentic → router → vLLM 경로가 Responses field/event를 보존하는가?

## 2. 먼저 환경을 고정한다

기록할 항목:

```text
Kubernetes version
kube-proxy mode: iptables / nftables / IPVS / absent(eBPF)
CNI and version
HTTP version per hop
Gateway/Ingress implementation and version
client connection pool settings
Service sessionAffinity/internalTrafficPolicy
router/image digest and source commit
vLLM version and model
```

같은 `Service` YAML도 dataplane implementation에 따라 관측 command와 rule shape가 다르다.

## 3. Service → EndpointSlice 확인

```bash
kubectl get service model-a -o yaml
kubectl get endpointslice -l kubernetes.io/service-name=model-a -o yaml
kubectl get pod -l app=model-a -o wide --show-labels
```

확인 항목:

- Service selector와 Pod label 일치
- EndpointSlice address와 Pod IP 일치
- port/targetPort
- `ready`, `serving`, `terminating`
- node/zone 정보
- endpoint 변화 timestamp

watch로 control-plane 전파를 본다.

```bash
kubectl get endpointslice \
  -l kubernetes.io/service-name=model-a \
  --watch -o wide
```

## 4. kube-proxy/conntrack 확인

iptables mode node에서 대표적으로 다음을 본다. root 권한과 cluster 정책을 지킨다.

```bash
iptables-save -t nat | rg 'KUBE-SERVICES|KUBE-SVC|KUBE-SEP'
conntrack -L -p tcp | rg '<cluster-ip>|<pod-ip>|<service-port>'
ss -ntp
```

nftables/IPVS/eBPF dataplane이면 해당 구현의 native command를 사용한다. chain name만 보고 mode를 단정하지 않는다.

packet capture는 client node, router Pod, selected backend에서 동일 request ID/flow를 연결한다.

```bash
tcpdump -nn -i any 'tcp port 8000'
```

민감한 prompt/body가 capture에 남을 수 있으므로 synthetic payload만 사용하고 pcap 보존 정책을 정한다.

## 5. keep-alive가 만드는 stickiness 실험

backend가 응답 header 또는 body에 자신의 Pod UID를 넣는 test endpoint를 준비한다.

### 실험 A: connection 매번 새로 생성

- request마다 client process 또는 connection을 새로 만든다.
- Pod UID 분포와 source port 변화를 기록한다.

### 실험 B: HTTP/1.1 keep-alive 하나 재사용

- 같은 client connection으로 request를 반복한다.
- request 수와 backend selection 수를 비교한다.

### 실험 C: HTTP/2 multiplexing

- 한 H2 connection에 concurrent stream을 보낸다.
- 모든 stream이 동일 backend로 가는지 확인한다.

판정 표:

| metric | 기대 |
|---|---|
| HTTP requests | A/B/C 모두 동일하게 많음 |
| TCP connections | A가 가장 많고 B/C가 적음 |
| Service backend selections | TCP connection 생성 수에 가까움 |
| Pod load uniformity | connection lifetime/request cost에 따라 달라짐 |

## 6. Endpoint condition/removal과 drain 실험

readiness 변화와 selector membership 제거를 한 실험으로 섞지 않고 두 케이스로 나눈다.

### Case A — readiness=false

1. Pod A로 long SSE 또는 WebSocket connection을 연다.
2. 별도 keep-alive HTTP connection도 연다.
3. Pod A readiness를 false로 만든다.
4. EndpointSlice에서 endpoint가 남아 있는지와 `conditions.ready: false` 전환 시각을 기록한다.
5. new connection이 Pod A를 피하기 시작하는 시각을 기록한다.
6. existing SSE/WS/keep-alive traffic이 계속 Pod A를 사용하는지 확인한다.

### Case B — Service selector membership 제거

1. 동일한 long-lived traffic을 만든다.
2. Service selector와 match하던 routing label을 Pod A에서 제거한다.
3. 해당 Service의 EndpointSlice membership에서 Pod A가 제거되는 시각을 기록한다.
4. new connection이 Pod A를 피하기 시작하는 시각을 기록한다.
5. existing connection이 유지되는지 확인한다.
6. application drain/Pod termination을 실행하고 close/reset 형태를 기록한다.

필수 timestamp:

```text
t0 readiness 또는 selector-label change
t1 EndpointSlice condition/membership update observed
t2 kube-proxy/dataplane rule update observed
t3 last new connection assigned to Pod A
t4 existing stream terminal/close
t5 Pod process exit
```

이 실험은 “readiness=false면 endpoint가 즉시 사라진다”와 “EndpointSlice membership에서 제거되면 기존 connection도 즉시 이동한다”라는 서로 다른 오해를 각각 검증한다.

## 7. SSE fragmentation proxy test

test proxy가 upstream byte stream을 다음 mode로 변환하게 한다.

- 1 byte씩 yield
- 임의 크기로 split
- 여러 event를 합쳐 yield
- UTF-8 character 중간 split
- event 사이 delay 삽입
- terminal event 직전 disconnect

client parser가 동일 semantic event sequence를 만드는지 golden result와 비교한다. TCP packet capture의 segment 수가 아니라 **decoded event sequence**가 assertion 대상이다.

## 8. Responses end-to-end fidelity matrix

각 path를 direct baseline과 비교한다.

```text
A. Client -> vLLM
B. Client -> model router -> vLLM
C. Client -> Agentic -> model router -> vLLM
D. Client -> Gateway -> Agentic -> model router -> vLLM
```

| 영역 | test case |
|---|---|
| request body | unknown/optional field, tools, reasoning, metadata, large input |
| model routing | valid model, alias off/on, unknown model, unavailable backend |
| non-stream | status, headers, complete JSON, usage, error body |
| SSE | event type/order/count, fragmentation, final event, content type |
| WebSocket edge | Upgrade, multi-turn, fallback, reconnect, close code |
| state | `previous_response_id`, failover to another Agentic replica, branch/parallel turn |
| tools | function/custom call, partial arguments, output continuation, duplicate retry prevention |
| P/D | `max_output_tokens`, KV transfer fields, prefill/decode error, SSE output |

body equality는 JSON key order/whitespace가 아니라 semantic JSON과 required raw extension preservation을 본다. 반대로 byte-transparent proxy를 계약으로 삼았다면 raw-body hash도 별도로 측정한다.

## 9. Router 선택권 검증

router log에 `decision_id`, selected Pod UID/IP, model, cache/load score를 남기고 backend가 받은 request에 같은 decision ID를 연결한다.

```text
router_selected_endpoint == actual_backend_endpoint
```

이 equality가 깨지면 중간 ClusterIP/DNS/pool이 selection을 다시 수행한 것이다. prompt/raw text는 log에 남기지 않고 hash, token count, cache hit length 같은 최소 정보만 사용한다.

## 10. 필수 metric

### transport

- active/new/closed TCP and WebSocket connections
- HTTP requests per connection
- HTTP version per hop
- first upstream byte / first complete event / terminal latency
- buffered bytes, slow-consumer count
- disconnect source와 close/reset code

### Kubernetes routing

- EndpointSlice endpoint count와 condition
- endpoint update-to-dataplane convergence
- new connections per backend
- existing connection age per backend
- drain duration와 forced termination 수

### LLM routing/state

- selected model/engine, selection reason
- queue/load score와 KV matched tokens
- expected vs actual backend
- KV hit/miss와 prefill tokens/latency
- hydration items/bytes/latency
- DB transaction/failure
- partial stream 뒤 retry attempt
- tool round와 idempotency conflict

## 11. 장애 주입 gate

- Agentic replica kill 후 다른 replica에서 continuation
- router restart 중 SSE와 new request
- selected vLLM Pod가 dispatch 직전 terminating
- EndpointSlice watch disconnect/relist
- DB failover와 commit ambiguity
- SSE partial delivery 후 outer proxy retry 차단
- WebSocket abnormal close 후 fallback/reconnect
- P/D prefill 성공·decode 실패, KV transfer timeout
- rolling update 중 long SSE/WS drain
- connection pool이 removed Pod를 계속 쓰는지 확인

production gate는 success rate 하나가 아니라 correctness, duplicate side effect, state recovery, latency degradation을 분리한다.
