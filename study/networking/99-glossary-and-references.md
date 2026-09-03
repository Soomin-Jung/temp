# 용어집과 공식 자료

업데이트: 2026-09-02

## 용어집

| 용어 | 의미 |
|---|---|
| application event | Responses delta/item lifecycle처럼 API가 정의한 semantic 단위 |
| backpressure | downstream 소비 속도가 느릴 때 upstream 전송/생성을 늦추는 압력 |
| ClusterIP | cluster 내부에서 Service를 나타내는 virtual IP |
| connection pool | destination별 TCP/HTTP connection을 재사용하는 client 구조 |
| conntrack | kernel이 network flow와 NAT state를 추적하는 기능 |
| DNAT | destination address/port를 다른 endpoint로 변환 |
| EndpointSlice | Service가 가리키는 network endpoint와 condition을 나타내는 Kubernetes API |
| full-duplex | 양쪽이 독립적으로 동시에 message를 전송 가능 |
| HTTP stream | HTTP/2/3에서 multiplexed request/response 단위. SSE stream과 혼동하지 않는다. |
| idempotency | 동일 operation을 여러 번 요청해도 외부 효과가 한 번과 동일하도록 하는 성질/계약 |
| L4 routing | IP/port/protocol/flow 중심 routing |
| L7 routing | HTTP/application 정보 중심 routing |
| LLM delta | text, reasoning, tool argument 등의 application-level 증분 |
| persistent connection | 여러 request 또는 긴 session에 재사용되는 transport connection |
| rehydration | response/conversation ID가 참조하는 typed state를 full inference input으로 재구성 |
| SSE | `text/event-stream` 형식으로 HTTP response에 server event를 연속 전달 |
| terminal event | operation이 completed/failed/cancelled임을 확정하는 마지막 application event |
| WebSocket | HTTP handshake 뒤 사용하는 full-duplex message-oriented protocol |

## HTTP와 streaming

- [RFC 9110: HTTP Semantics](https://www.rfc-editor.org/rfc/rfc9110.html)
- [RFC 9112: HTTP/1.1](https://www.rfc-editor.org/rfc/rfc9112.html)
- [RFC 9113: HTTP/2](https://www.rfc-editor.org/rfc/rfc9113.html)
- [RFC 9114: HTTP/3](https://www.rfc-editor.org/rfc/rfc9114.html)
- [WHATWG: Server-sent events](https://html.spec.whatwg.org/multipage/server-sent-events.html)
- [RFC 6455: The WebSocket Protocol](https://www.rfc-editor.org/rfc/rfc6455.html)
- [RFC 8441: Bootstrapping WebSockets with HTTP/2](https://www.rfc-editor.org/rfc/rfc8441.html)

## Kubernetes networking

- [Kubernetes Service](https://kubernetes.io/docs/concepts/services-networking/service/)
- [Kubernetes EndpointSlices](https://kubernetes.io/docs/concepts/services-networking/endpoint-slices/)
- [Virtual IPs and Service Proxies](https://kubernetes.io/docs/reference/networking/virtual-ips/)
- [Service Internal Traffic Policy](https://kubernetes.io/docs/concepts/services-networking/service-traffic-policy/)
- [Topology Aware Routing](https://kubernetes.io/docs/concepts/services-networking/topology-aware-routing/)
- [Gateway API HTTP routing](https://gateway-api.sigs.k8s.io/guides/user-guides/http-routing/)
- [Gateway API Inference Extension](https://gateway-api-inference-extension.sigs.k8s.io/)

## LLM serving/router source references

- [vLLM Production Stack](https://github.com/vllm-project/production-stack)
- [vLLM Production Stack 0.1.9 release](https://github.com/vllm-project/production-stack/releases/tag/vllm-stack-0.1.9)
- [production-stack PR #691: `/v1/responses` route](https://github.com/vllm-project/production-stack/pull/691)
- [vLLM Router](https://github.com/vllm-project/router)
- [vLLM Agentic API](https://github.com/vllm-project/agentic-api)

공식 문서만으로 implementation-specific connection pool, CNI, router behavior가 확정되지는 않는다. 배포 image digest와 source commit을 연결하고 [관측·실험 문서](12-observability-and-labs.md)의 방법으로 실제 path를 검증한다.
