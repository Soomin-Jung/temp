# Buzz HPC bundled Helm deployment

업데이트: 2026-10-19 KST  
상태: **소스·차트 정적 검토 완료 / HPC 실제 설치 미검증**  
대상: [block/buzz](https://github.com/block/buzz) commit [3debf18f1448334aa7bc202d328d93bb9651e168](https://github.com/block/buzz/tree/3debf18f1448334aa7bc202d328d93bb9651e168), chart `deploy/charts/buzz` 0.1.11.

## 권장 HPC 사용 경로: Linux X 세션 + Local Agent

- **[NFS 공용 Buzz Desktop 설치·`source env.sh; buzz` 실행 가이드](local-agent-nfs-desktop.md)** — Linux X 서버에서 Local Agent 실행, 사내 vLLM·GHES 연결, 사용자별 키·상태 격리 및 OS/glibc/FUSE/NFS 점검.
- **기본값은 Local Agent**이며, 사용자가 Kubernetes Pod를 생성할 필요가 없다. 공식 Helm 차트는 Relay/저장소 계층을 설치하며, Remote Agent용 Kubernetes Provider는 선택 경로다.
- 이 가이드는 Desktop이 아닌 웹 브라우저로 모든 기능을 사용한다는 이전 요구사항과 **별개의 운영 옵션**이다. 기업 OIDC SSO·GHES 네이티브 UI의 제약은 동일하게 남는다.
- **호환성 차단 조건**: 실제 X 서버에서 Buzz Linux AppImage의 glibc, GLib, FUSE, NFS `exec` 지원을 먼저 확인한다. Desktop 릴리스와 이 문서의 Relay 소스 검토 커밋이 서로 다를 수 있다.

## 도메인 및 GitHub Enterprise 연동

- [Buzz Relay 내부 도메인과 Ingress](domain-and-ingress.md)
- [GitHub Enterprise Server 연동 검토](github-enterprise-integration.md)
- [Ingress values 예시](values-ingress.example.yaml)

- [Web-only, GHES 서버 자동화, OIDC 지원 범위](web-oidc-ghes-audit.md) — 공식 구현과 별도 개발 범위의 기술 감사

## 1. 배포 결정

Buzz 공식 Helm 차트의 **Quickstart / evaluation 구성**을 이용하여 Buzz Relay, PostgreSQL, Redis, S3 호환 Silo를 **한 Helm Release**에 설치한다. `quickstart: true`는 프로파일 표시용으로, 실제 설치 여부는 `postgresql.enabled`, `redis.enabled`, `minio.enabled`를 각각 `true`로 설정해야 한다. Silo는 MinIO API 호환 오브젝트 스토리지로, 공식 차트는 여전히 `minio`라는 키와 Service 이름을 사용한다.

- 설정 예시: [values-bundled.yaml](values-bundled.yaml) (원본 레지스트리 사용), [values-mirrored.example.yaml](values-mirrored.example.yaml) (내부 레지스트리 사용).
- **정확한 이미지와 추가 조건**: [image-manifest.md](image-manifest.md).
- **중요**: 이 구성을 그대로 Production HA로 간주하지 않는다. 번들 PostgreSQL/Redis/Silo는 평가용 단일 인스턴스이며, Silo Pod와 데이터 볼륨에도 별도 복구 전략이 필요하다. 추후 운영 전환에서는 HA 스토리지 및 명시적 Secret 관리 방식으로 재설계한다.
- **Cargo/Node 빌드 불필요**: 공식 Buzz 컨테이너 이미지에 `buzz-relay`, `buzz-admin`, `buzz-pair-relay`, 사용자/관리자 Web UI가 포함되어 있다. `.cargo/config.toml`은 원본 소스 **이미지 빌드 시에만** 사용한다. Helm 차트에 마운트하지 않는다.

## 2. 리소스 토폴로지

| 항목 | Kubernetes 리소스 | 이미지 | 저장 영역 |
| --- | --- | --- | --- |
| Buzz Relay | Deployment, Service | ghcr.io/block/buzz | Git scratch + pack cache (`emptyDir` 권장) |
| PostgreSQL | CloudPirates postgres Subchart의 StatefulSet/Service | postgres:18.4 | PVC |
| Redis | CloudPirates redis Subchart의 StatefulSet/Service | redis:8.8.0 | PVC |
| Silo | Buzz 차트 직접 생성 Deployment/Service | pgsty/silo:RELEASE.2026-09-16T00-00-00Z | PVC |
| Bucket 초기화 | Buzz 차트 `minio-init` Job | pgsty/mc:동일 release | 별도 영속 볼륨 없음 |
| Bucket 준비 대기 | Relay InitContainer (Silo 번들 활성 시) | pgsty/mc:동일 release | 임시 |

기본 프로파일에서는 **서로 다른 실행 이미지 5종**이고, 추가 서비스별 이미지를 직접 빌드할 필요는 없다. PostgreSQL/Redis 차트가 참조하는 `cloudpirates/common:2.2.0`은 **Helm library chart**이며 컨테이너 이미지가 아니다.

### 필수 값

- `relayUrl`: 외부 클라이언트가 연결하는 DNS 기반 `wss://` URL. 내부 Ingress/Proxy를 별도로 구성한다.
- `ownerPubkey`: `relay.requireRelayMembership=true`라면 실제 소유자 Nostr 공개키(64자리 소문자 hex). 데모 편의를 위해 membership을 끄면 릴레이가 열린 상태가 되므로 HPC 공용망에 권장하지 않는다.
- 세 저장 서비스는 차트에서 생성하는 연결 정보와 Secret을 사용한다. **번들 Silo 사용 시 `secrets.existingSecret`을 지정하면 공식 차트가 설치를 거부**한다.
- 저장소 `StorageClass`와 PVC Binding을 확인한다. Silo 자체 PVC는 이 차트에서 별도 storageClass 지정 옵션이 없으므로 **클러스터 기본 StorageClass**가 필요하다. PostgreSQL/Redis의 `persistence.storageClass`는 따로 지정할 수 있다.
- HTTPS/WSS 인증서, 사내 DNS, WebSocket 프록시 타임아웃(1시간 이상), 내부 네트워크 정책을 별도 준비한다.

## 3. 버전 고정과 차트 준비

검토한 Git HEAD의 `Chart.yaml`은 0.1.11이지만, 이는 **GHCR에 해당 버전이 게시되었다는 증거가 아니다**. 공식 README에는 공개 OCI 차트 0.1.10 예시가 남아 있다. **0.1.11을 쓰려면 소스 커밋을 고정하고 chart dependency를 수집하여 직접 패키징**한다.

외부 접근 가능한 Staging에서:

~~~bash
git clone https://github.com/block/buzz.git
cd buzz
git checkout 3debf18f1448334aa7bc202d328d93bb9651e168

# Helm용 차트 다운로드 (Cargo/npm 설치는 불필요)
helm dependency build deploy/charts/buzz
helm package deploy/charts/buzz -d dist

# 배포 전 Helm 차트 메타데이터와 실제 포함 파일 확인
helm show chart dist/buzz-0.1.11.tgz
tar tf dist/buzz-0.1.11.tgz | grep '^buzz/charts/'
~~~

`Chart.lock`은 PostgreSQL `0.19.5` / Redis `0.30.3`를 지정한다. 두 차트의 내부 의존성은 CloudPirates Common `2.2.0`. OCI 엔드포인트는 `registry-1.docker.io/cloudpirates`. **번들 기능을 `enabled=false`로 설정해도 `helm dependency build`는 선언된 하위 차트를 다운로드**한다. 패키징이 끝난 `.tgz`를 HPC에 가져오면 실행 환경에서 의존성을 다시 다운로드할 필요가 없다. OCI 패키지 다운로드를 시도하지 않도록 HPC에서는 `helm dependency build`를 재실행하지 않는다.

공식 Relay 이미지는 `ghcr.io/block/buzz`이며, Chart.AppVersion은 0.1.0으로 고정되어 있어서 **실제 게시 이미지 태그를 확인한 뒤 `image.tag`를 반드시 명시**해야 한다. GitHub Workflow는 main 푸시에서 `sha-<sha7>` 계열 태그, `relay-v*` 릴리스에서 버전 태그를 게시하도록 구성되어 있다. **차트 0.1.11과 Buzz Relay 태그는 독립 버전**이다.

## 4. 배포 (원본 레지스트리 또는 검증된 프록시)

다음 예시는 staging에서 `buzz-0.1.11.tgz`와 [values-bundled.yaml](values-bundled.yaml)을 반입했다는 전제다. 플레이스홀더를 실제 값으로 바꾼다.

~~~bash
export BUZZ_OWNER_PUBKEY='<실제 소유자의 64자리 소문자 hex>'
export BUZZ_RELAY_IMAGE_TAG='<GHCR에서 확인한 Buzz relay 이미지 태그>'
export BUZZ_RELAY_URL='wss://buzz.hpc.example.com'

test "${#BUZZ_OWNER_PUBKEY}" -eq 64
printf '%s' "$BUZZ_OWNER_PUBKEY" | grep -Eq '^[0-9a-f]{64}$'
test -n "$BUZZ_RELAY_IMAGE_TAG"

helm template buzz ./buzz-0.1.11.tgz \
  -n buzz -f values-bundled.yaml \
  --set-string ownerPubkey="$BUZZ_OWNER_PUBKEY" \
  --set-string relayUrl="$BUZZ_RELAY_URL" \
  --set-string image.tag="$BUZZ_RELAY_IMAGE_TAG" \
  | grep 'image:' | sort -u

# 주의: helm template 전체 출력에는 자동 생성된 Secret 데이터가 들어간다.
# 전체 manifest를 파일/CI 로그로 남기지 말 것.

helm upgrade --install buzz ./buzz-0.1.11.tgz \
  -n buzz --create-namespace \
  -f values-bundled.yaml \
  --set-string ownerPubkey="$BUZZ_OWNER_PUBKEY" \
  --set-string relayUrl="$BUZZ_RELAY_URL" \
  --set-string image.tag="$BUZZ_RELAY_IMAGE_TAG" \
  --wait --wait-for-jobs --timeout 15m
~~~

다른 릴리스 이름을 사용할 경우 Chart-managed Secret 및 PostgreSQL/Redis 연결 문자열 생성 이름이 달라질 수 있으므로, **`buzz` 릴리스 이름으로 시작**하고 반드시 `helm template` 결과를 확인한다.

## 5. 완전 내부 Registry 전환

- 원본 5종 이미지를 내부 Registry에 미러링하고 검증한 내부 태그/다이제스트로 [values-mirrored.example.yaml](values-mirrored.example.yaml)을 수정한다.
- `helm template -f values-bundled.yaml -f values-mirrored.example.yaml`로 **실제 Deployment / StatefulSet / Job / InitContainer 이미지**를 점검한다.
- PostgreSQL/Redis는 하위 차트의 `image.registry`, `image.repository`, `image.tag`를 조정한다. Silo/mc는 Buzz 차트의 문자열 값인 `minio.image`, `minio.mcImage`를 조정한다.
- Digest를 붙인 원본 이미지 참조를 임의로 수정하지 않는다. Registry 변경 후 미러에 존재하는 다이제스트인지 확인하고, 일치하지 않으면 **새 Registry의 실제 digest로 교체**한다.
- 사내 프록시를 통해 OCI Pull이 가능하면 반드시 미러링할 필요는 없지만, 노드 `containerd`의 인증서/프록시/인증 설정은 Pod 환경변수나 Cargo 설정과 별개다.
- `image.pullSecrets`는 Buzz 주 컨테이너용이며, 하위 DB/Redis는 필요하다면 해당 서브차트 image pull secrets를 별도로 설정한다. Silo/버킷 Init Job은 공식 템플릿에 별도 `imagePullSecrets` 옵션이 없으므로 **노드 전역 레지스트리 인증, namespace 기본 ServiceAccount 설정, 또는 템플릿 패치**로 해결해야 한다.

## 6. 사전 점검 / 배포 검증

~~~bash
kubectl -n buzz get pods,pvc,svc,jobs
kubectl -n buzz get pods -o jsonpath='{range .items[*]}{.metadata.name}{" => "}{range .spec.initContainers[*]}{.image}{" "}{end}{range .spec.containers[*]}{.image}{" "}{end}{"\n"}{end}'
kubectl -n buzz get events --sort-by=.lastTimestamp | tail -40

# Relay liveness/readiness (8080)
kubectl -n buzz port-forward svc/buzz 8080:8080
# 별도 터미널: curl -fsS http://127.0.0.1:8080/_readiness

# Pod / Job 장애 시
kubectl -n buzz describe pod <POD>
kubectl -n buzz logs <POD> -c relay --tail=200
kubectl -n buzz get jobs
~~~

릴레이는 초기 버킷이 준비될 때까지 `pgsty/mc` InitContainer에서 대기한다. S3 conformance probe가 기본 활성화되어 있으므로 잘못된 접근키·버킷·S3 구현은 Relay 기동을 실패시킬 수 있다. 운영 진입 전에는 **WebSocket 연결, 사용자 메시지 왕복, Redis fan-out, PostgreSQL 영속성, S3 업로드/조회, PVC 재기동**을 검증한다.

## 7. 운영 전환의 차단 조건

1. **GitOps 비호환:** Quickstart chart-managed Secret은 Helm `lookup` 기반이다. `helm template`/ArgoCD/Flux가 매번 새 값을 생성할 수 있다. 따라서 초기 PoC는 `helm install/upgrade` 방식만 사용한다.
2. **번들 Silo와 existingSecret 동시 사용 불가:** 공식 `quickstart-minio.yaml`에서 의도적으로 `fail` 처리한다. 운영용 불변 Secret + 번들 Silo를 함께 쓰려면 upstream 수정 또는 커스텀 차트가 필요하다.
3. **단일 장애점:** PostgreSQL, Redis, Silo는 평가용 single instance. PVC가 있어도 HA/백업을 대체하지 않는다.
4. **릴레이 개인키와 데이터:** `BUZZ_RELAY_PRIVATE_KEY`는 릴레이 Identity이다. 차트 Secret/DB/PVC/S3 객체와 별도로 백업·복원 정책을 검증한다.
5. **MinIO와 Silo 버전 호환:** 공식 차트는 Silo fork를 사용한다. 저장 볼륨 교체 시 기존 MinIO 이미지와의 데이터 호환 및 업그레이드 절차를 독립 검증한다.
6. **HPC Egress:** 이미지 Pull 성공은 런타임 외부 HTTP 요청 허용을 의미하지 않는다. Push는 기본 비활성, KLIPY Key 미설정; Agent와 외부 Integration은 추가 점검한다.

## 8. Upstream evidence

- [Buzz Chart.yaml](https://github.com/block/buzz/blob/3debf18f1448334aa7bc202d328d93bb9651e168/deploy/charts/buzz/Chart.yaml), [Chart.lock](https://github.com/block/buzz/blob/3debf18f1448334aa7bc202d328d93bb9651e168/deploy/charts/buzz/Chart.lock), [values.yaml](https://github.com/block/buzz/blob/3debf18f1448334aa7bc202d328d93bb9651e168/deploy/charts/buzz/values.yaml)
- [Buzz Quickstart and Secret limitations](https://github.com/block/buzz/blob/3debf18f1448334aa7bc202d328d93bb9651e168/deploy/charts/buzz/README.md), [Silo template](https://github.com/block/buzz/blob/3debf18f1448334aa7bc202d328d93bb9651e168/deploy/charts/buzz/templates/quickstart-minio.yaml)
- [Postgres chart 0.19.5](https://github.com/CloudPirates-io/helm-charts/blob/postgres-0.19.5/charts/postgres/values.yaml), [Redis chart 0.30.3](https://github.com/CloudPirates-io/helm-charts/blob/redis-0.30.3/charts/redis/values.yaml)
- [Buzz multi-stage Dockerfile](https://github.com/block/buzz/blob/3debf18f1448334aa7bc202d328d93bb9651e168/Dockerfile), [Relay image publication workflow](https://github.com/block/buzz/blob/3debf18f1448334aa7bc202d328d93bb9651e168/.github/workflows/docker.yml)
