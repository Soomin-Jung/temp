# HPC Prometheus selective metric sharing

작성일: 2026-10-19 | 목적: HPC망의 내부 Prometheus 전체를 공개하지 않고, 타 부서에는 승인된 H200 DCGM 및 (추후 승인 시) vLLM 메트릭만 제공한다.

> **상태/출처**: `source-scrape-values.reference.yaml`은 [vllm-production-stack-custom PR #7](https://github.com/Soomin-Jung/vllm-production-stack-custom/pull/7)의 commit `c86d1736640ed2ac640e7e379264dfb548391767`에서 **원문 그대로** 복사했다. 해당 PR은 **미병합·종료**되었으므로 실제 HPC 운영 설정임이 검증된 것은 아니다. `source-scrape-values.reference.yaml`은 운영에 바로 적용하는 파일이 아닌 참조 스냅샷이다.

## 파일 구성

| 파일 | 용도 |
|---|---|
| `source-scrape-values.reference.yaml` | 기존 DCGM / standalone vLLM / P-D engine / P-D router / LiteLLM 스크랩 구성을 담은 미병합 PR 스냅샷 |
| `federation-h200-only.values.yaml` | 초기 승인 범위인 H200 DCGM 전용 신규 Prometheus Helm values |
| `federation-h200-vllm.values.yaml` | vLLM 공개가 추가 승인된 뒤 쓰는 **별도 Federation Job 2개** Helm values 예시 |
| `federation-one-job.prometheus.yml` | `match[]`를 여러 개 지정해 **단일 Federation Job**으로 DCGM과 vLLM을 함께 수집하는 Prometheus 원시 설정 |
| `validation.md` | 배포 전후 검증, 정보 노출 점검, 과거 조회 테스트 |

## 원본 scrape 구조 (출처: PR #7)

- `nvidia-dcgm-metrics`: `kube-system`의 dcgm-exporter Pod `:9200`. `node`, `instance` 및 `gpu` 리라벨.
- `kubernetes-vllm-instances`: `inference`, `test`에서 일반 vLLM 컨테이너 `:8000`, P-D router 제외.
- `kubernetes-vllm-pd-engines`: `inference`, `test`; `pd-cell=true` 및 `(prefill|decode)-*` 컨테이너/포트명.
- `kubernetes-vllm-pd-router`: `router-metrics` 포트(기준 `:29000`).
- `kubernetes-litellm-proxy-instances`: `:4000`.

위 목록은 **소스 설정에 정의된 내용**이지, 실제 클러스터에서의 활성 Target 목록이라는 뜻이 아니다. 먼저 운영 Prometheus의 `/api/v1/targets`, `/api/v1/status/config`와 대조한다.

## Federation 동작/보안 경계

```text
DCGM Exporter / vLLM / Kubernetes / HW
          |
          | 원본 Prometheus가 평소처럼 scrape
          v
[기존 HPC Prometheus: 전체 시계열 TSDB] -- /federate (내부 접근만 허용)
          ^
          | 신규 Prometheus가 지정한 주기로 GET /federate?match[]=...
          |
[신규 공개 전용 Prometheus: 허용 샘플만 별도 TSDB]
          ^
          | 인증·네트워크 제어된 조회 API
          |
       [타 부서]
```

- 원본 Prometheus에는 별도의 Federation 활성화 플래그가 필요하지 않다. `/federate` API가 기본 제공된다.
- `/federate`의 `match[]`는 **현재 시계열에 대한 instant-vector selector**다. 같은 Job에 `match[]`를 여러 번 제공하면 결과는 **합집합(OR)**이다.
- `match[]` 자체는 원본의 RBAC/보안 정책이 **아니다**. 원본 URL 및 `/federate`를 타 부서에 절대 공개하지 않는다.
- 신규 Prometheus의 `metrics_path: /federate`도 Prometheus의 일반 **scrape**다. 대상 exporter를 다시 직접 스크랩하는 게 아니라 기존 Prometheus의 최신 값들을 주기적으로 가져와 저장한다.
- `honor_labels: true`는 원본 `job`, `instance` 등을 보존하는 용도다.
- 원본에서 만들어진 메트릭 이름, 라벨, GPU 모델이 운영 구성과 다를 수 있다. `modelName=~".*H200.*"`가 맞는지 사전 검증한다. **`modelName` 없는 시계열은 아래 H200 정책으로 수집되지 않는다.**
- 예시의 DCGM 메트릭은 허용 목록 예시일 뿐 **최종 승인된 데이터 목록을 뜻하지 않는다**. `DCGM_.*`를 무조건 공개하는 식의 광범위한 정규식은 피한다.
- vLLM Job은 공개가 별도 승인될 때만 활성화한다. `APPROVED_MODEL`을 실제 승인된 `model` 라벨 값으로 바꿔야 동작한다.

## 여러 개의 metric selector / Job 구성

### A. 단일 Job + 여러 `match[]` (합집합)

```yaml
params:
  'match[]':
    - '{job="nvidia-dcgm-metrics",modelName=~".*H200.*",__name__=~"DCGM_FI_DEV_(GPU_UTIL|FB_USED|FB_FREE|POWER_USAGE|GPU_TEMP)"}'
    - '{job="kubernetes-vllm-instances",namespace="inference",model="APPROVED_MODEL",__name__=~"vllm:(num_requests_running|num_requests_waiting|prompt_tokens_total|generation_tokens_total)"}'
    - '{job="kubernetes-vllm-pd-engines",namespace="inference",model="APPROVED_MODEL",__name__=~"vllm:(num_requests_running|num_requests_waiting|prompt_tokens_total|generation_tokens_total)"}'
```

하나의 `job_name`, `scrape_interval`, `metrics_path`, target 설정을 공유한다. 검증된 원시 설정은 `federation-one-job.prometheus.yml`에 있다. 실수 방지를 위해 예시에서 `metric_relabel_configs`도 Job/namespace/model/metric 이름의 허용 조건을 **하나의 OR 정규식**으로 재검증한다. 별도의 `action: keep`를 2개 쓰면 AND가 되어 의도와 다르게 샘플이 전부 제거된다.

### B. 2개 Job (권장)

- `federate-h200-dcgm`: 원본 `job=nvidia-dcgm-metrics`, H200 `modelName`, 허용 DCGM 메트릭 이름, **60초**.
- `federate-approved-vllm`: 원본 standalone/P-D engine `job`, `namespace=inference`, `model=APPROVED_MODEL`, 허용 `vllm:` 이름, **15초**.

원본 주소가 같아도 Job별 `params.match[]`, `scrape_interval`, `metric_relabel_configs`를 다르게 지정할 수 있다. 목적과 성능/카디널리티 정책이 다른 DCGM/vLLM은 분리하는 편이 운영에 유리하다.

## 수집 간격과 과거 PromQL 조회 (핵심)

**Federation은 원본에서 최신 샘플만 가져오며, 원본의 과거 TSDB 데이터를 소급 복제하지 않는다.**

| 질의 | 동작 |
|---|---|
| 신규 Prometheus 시작 후 30분이 지났고 최근 15분 `query_range` 조회 | 신규 TSDB에 저장된 구간에 한하여 정상 조회 |
| 시작 30분 후 `avg_over_time(metric[24h])` | 최대 30분 정도의 신규 저장 샘플로 계산; 24시간 전체를 원본에 요청하지 않음 |
| 시작 30분 후 7일 전 시점의 `query`/`query_range` | 신규 TSDB에는 데이터가 없어 결과 없음 |
| 원본 Prometheus의 30일 이력 | Federation으로 백필되지 않음 |
| 원본 15초 scrape, 신규 60초 federation | 대체로 60초당 최신 샘플만 저장됨. 원본의 모든 15초 샘플이 전송되는 것은 아님 |
| 원본 60초 scrape, 신규 15초 federation | 같은 원본 샘플을 반복 조회할 수 있으며 원본 해상도가 높아지지 않음 |

`query`, `query_range`가 신규 Prometheus에 들어왔다고 원본 Prometheus로 **자동 프록시/포워딩하지는 않는다**. 질의는 신규 TSDB를 대상으로 수행된다. `honor_timestamps` 기본 동작을 유지하면 federation 값에는 원본 샘플 타임스탬프가 보존될 수 있다.

과거 원본 이력까지 즉시 필요하다면 **별도 요구사항**으로 판단한다.
- `remote_read`: 원본의 `/api/v1/read`를 통해 원시 과거 시계열을 요청 가능하지만, 보호된 원본에 직접 연결하면 비허용 시계열도 읽을 가능성이 있어 **이번 공개 전용 구성에서 직접 연결을 권장하지 않는다**. 필터링 전용 게이트웨이/권한 통제가 따로 필요.
- 허용 데이터만 검증해 신규 TSDB로 **선별 백필**하거나, 인증/권한/셀렉터 강제를 갖춘 별도 Query Proxy를 구축하는 대안 검토.
- Prometheus 자체 보존 기간은 `server.retention` 및 PVC 구성에 의존한다.

메트릭 성격별로 간격은 정해야 한다. DCGM 상태 지표 60초, vLLM latency/rate/short-burst 분석 15초는 **검증 출발점**일 뿐 필수값은 아니다. 원본 scrape 간격과 실제 요구 SLA를 맞춘다.

## 차트/배포 조건

이 예시는 `prometheus-community/prometheus` **단독 서버 Helm 차트**를 가정한다.

- Helm chart **v28 이후**에는 기본 scrape 구성이 `scrapeConfigs`로 옮겨졌다. 본 문서는 PR #7과 호환되게 `serverFiles.prometheus.yml.scrape_configs`를 사용하므로 **`scrapeConfigs: null`** 로 기본 scrape Job들을 제거한다.
- `kube-prometheus-stack`(Prometheus Operator)에서는 `prometheus.prometheusSpec.additionalScrapeConfigs` 등 별도 설정 경로가 필요하며 파일을 그대로 넣으면 안 된다.
- 샘플 values의 원본 주소는 실제 클러스터 서비스 DNS로 수정해야 한다. 외부 Ingress, 인증/TLS, 공개 API allowlist, NetworkPolicy는 **별도 설정 후** 오픈한다.
- 기본 `up` 등 자동 생성 메트릭, `/api/v1/targets`, `/api/v1/status/config`, `/api/v1/labels`, `/api/v1/series`의 정보 노출도 별도 점검해야 한다. `metric_relabel_configs`는 자동 `up` 메트릭에 적용되지 않는다.
- `instance`, `node`, `gpu`, `UUID`, `pod`, `container` 등 식별자 공개 가능 여부를 승인받아야 한다. 라벨을 임의로 삭제하면 서로 다른 GPU/Pod 시계열이 충돌할 수 있으므로 라벨 익명화 설계를 별도로 한다.
- `model`은 모델 이름이지 GPU 타입이 아니다. vLLM이 **H200 노드에서 실행되었는지**까지 제한하려면 소스 시계열에 신뢰할 수 있는 node/hardware 매핑 라벨을 더 확보해야 한다.

## 공식 자료

- [Prometheus federation](https://prometheus.io/docs/prometheus/latest/federation/)
- [Prometheus scrape configuration / relabeling](https://prometheus.io/docs/prometheus/latest/configuration/configuration/)
- [Prometheus remote read](https://prometheus.io/docs/prometheus/latest/querying/remote_read_api/)
- [Prometheus storage](https://prometheus.io/docs/prometheus/latest/storage/)
- [prometheus-community/prometheus Helm chart](https://github.com/prometheus-community/helm-charts/tree/main/charts/prometheus)
