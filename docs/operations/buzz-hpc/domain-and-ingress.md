# Buzz HPC 내부 도메인 / HTTPS / WSS

업데이트: 2026-10-19. 소스: block/buzz 3debf18f, Helm chart source 0.1.11.

## 결정

- **인터넷 공인 도메인은 불필요**하고 내부 DNS와 TLS 인증서를 가진 FQDN을 권장한다.
- 공식 Helm 차트의 `relayUrl`은 필수 값이다. 없으면 chart template 검증에 실패한다.
- `relayUrl: wss://buzz.example.internal`이 Relay WebSocket 주소, 기본 Ingress Host, 기본 Media URL의 기준이다.
- 내부 사용자의 Buzz Desktop 또는 Browser가 Relay FQDN에 접근하고 사내 CA를 신뢰해야 한다.
- Ingress 앞단의 WebSocket idle timeout(read/send)을 1시간 이상 확보한다.
- GitHub Enterprise Server의 Hostname은 Buzz Relay Hostname과 별개이며 GHES를 Buzz Relay가 자동 Proxy하지 않는다.

## 검증용 대안

- 단일 작업자 테스트: `kubectl -n buzz port-forward svc/buzz 3000:3000` 후 클라이언트에서 `ws://localhost:3000`.
- 사내 IP + `ws://IP:PORT`: 단기 테스트는 가능하지만 TLS/브라우저 혼합 콘텐츠와 장기 URL 안정성에 주의한다.
- 공유 PoC: 내부 FQDN + WSS + Ingress 사용.

## Ingress

[values-ingress.example.yaml](values-ingress.example.yaml)을 [values-bundled.yaml](values-bundled.yaml) 뒤에 적용한다. 샘플 hostname과 TLS Secret은 실환경에 맞게 교체한다.

```bash
helm upgrade --install buzz ./buzz-0.1.11.tgz -n buzz --create-namespace \
  -f values-bundled.yaml -f values-ingress.example.yaml \
  --set-string ownerPubkey="$BUZZ_OWNER_PUBKEY" \
  --set-string image.tag="$BUZZ_RELAY_IMAGE_TAG" \
  --wait --wait-for-jobs --timeout 15m
```

추가 검증: 사내 DNS, Nginx Ingress Controller, TLS Secret, WebSocket Upgrade, Media /media URL, Git /git URL, Client 사내 CA Trust. RelayUrl만 설정해도 자동 DNS나 인증서가 생성되는 것은 아니다.

## 소스 근거

- [Helm required relayUrl validation](https://github.com/block/buzz/blob/3debf18f1448334aa7bc202d328d93bb9651e168/deploy/charts/buzz/templates/_validate.tpl)
- [Ingress template](https://github.com/block/buzz/blob/3debf18f1448334aa7bc202d328d93bb9651e168/deploy/charts/buzz/templates/ingress.yaml)
- [Default Media URL logic](https://github.com/block/buzz/blob/3debf18f1448334aa7bc202d328d93bb9651e168/deploy/charts/buzz/templates/_helpers.tpl)
- [Official NOTES / WebSocket timeout](https://github.com/block/buzz/blob/3debf18f1448334aa7bc202d328d93bb9651e168/deploy/charts/buzz/templates/NOTES.txt)
