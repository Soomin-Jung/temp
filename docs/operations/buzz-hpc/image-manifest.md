# Buzz HPC — image & Helm dependency manifest

검토 소스: block/buzz commit 3debf18f1448334aa7bc202d328d93bb9651e168 / Buzz Chart 0.1.11.  
**실제 게시 이미지 태그 및 원격 Pull 가능 여부는 확인되지 않았으며, 미러/실제 배포 전에 Pull과 Digest를 검증할 것.**

## A. 기본 Quickstart 컨테이너 이미지: 5종

| Source image reference | 배포 위치 | 현재 소스에서 확인된 값 |
| --- | --- | --- |
| ghcr.io/block/buzz:<verified-relay-tag> | Relay (필수), 선택 기능 Pairing/Operator Job | 차트 image.tag 기본값은 비어 있으며 Chart.AppVersion 0.1.0으로 fallback하므로 게시된 태그를 명시 |
| docker.io/library/postgres:18.4@sha256:8ff36f3c66371cba71d20ceedccfc3de9669a68737607888c4ef0af93abe8e39 | CloudPirates PostgreSQL | postgres chart 0.19.5 기본값 |
| docker.io/library/redis:8.8.0@sha256:aa049e689e141a4358ad1d4562dc49c88a89fbab711fd8fcc33f684c80b26301 | CloudPirates Redis | redis chart 0.30.3 기본값 |
| docker.io/pgsty/silo:RELEASE.2026-09-16T00-00-00Z@sha256:635197cb9f36d01bee221d34d1c7d7960f6a95c48b0b6c01d99cd13bdae51a46 | S3 object server | Buzz minio.image 기본값 |
| docker.io/pgsty/mc:RELEASE.2026-09-16T00-00-00Z@sha256:cfc83108c3abb371f8fb84d99c1fdc88f8c237e022409b0081fb7c0a3be634dd | Bucket init Job + Relay initContainer | Buzz minio.mcImage 기본값 |

**이미지 갯수 판단 조건:** Redis Sentinel / Metrics exporter, PostgreSQL Metrics exporter, volumePermissions initContainer 등은 본 가이드 기본값에서 비활성이다. 특정 추가 기능을 켜면 추가 이미지를 `helm template`에서 재확인해야 한다. CloudPirates chart는 upstream 수정 시 이미지 기본값도 달라질 수 있으므로 반드시 잠금 버전을 유지한다.

### Image pulling과 빌드의 구분

- K8s 노드 image pulling: ghcr.io, docker.io 또는 내부 프록시/미러에서 **완성된 OCI 이미지**를 가져온다. Cargo Registry는 사용하지 않는다.
- 처음부터 Rust/Web UI 소스를 다시 빌드할 때만 Rust Cargo, npm, apt, Git 의존성이 발생한다. Runtime/Chart에는 Cargo 설정 마운트가 필요 없다.
- Bundle Silo uses `pgsty/silo`, `pgsty/mc`: 이름만 MinIO-compatible. 기존 MinIO 이미지로 마음대로 치환하지 않는다.

## B. Helm OCI 차트 의존성 (이미지가 아님)

| Artifact | Version | Source |
| --- | --- | --- |
| Buzz chart | 0.1.11 (검토한 Git 소스) | deploy/charts/buzz |
| CloudPirates postgres | 0.19.5 | oci://registry-1.docker.io/cloudpirates/postgres |
| CloudPirates redis | 0.30.3 | oci://registry-1.docker.io/cloudpirates/redis |
| CloudPirates common library | 2.2.0 | oci://registry-1.docker.io/cloudpirates/common |

CloudPirates common library chart는 postgres/redis의 하위 의존성이다. `helm dependency build`와 `helm package`는 외부 Staging에서 한 번 수행하고 의존성이 포함된 차트를 내부에 반입한다. 테스트용 `helm template`에서는 **모든 리소스의 image 필드**를 확인하고 그 출력을 이미지 Pull 승인 목록의 최종 근거로 사용한다.

## C. 내부 Registry 이미지 전환

예상 내부 경로는 [values-mirrored.example.yaml](values-mirrored.example.yaml)을 참고한다. 내부 호스트명은 샘플이고 실제 생성된 주소가 아니다. Docker Hub Official Library 이미지에 대해서는 원본 `postgres`/`redis`를 `library/postgres`/`library/redis`로 정규화했다.

Staging에서 `skopeo copy --all docker://<SOURCE_REF> docker://<INTERNAL_REF>` 등을 사용해 아키텍처별 OCI 이미지를 복사하고, HPC Registry의 실재 이미지 태그/Digest를 검증한다. 다중 아키텍처 Manifest가 필요 없는 경우에도 CPU 아키텍처 일치를 확인한다. **원본 Digest를 내부 경로로 복사했다고 자동으로 동일하다고 가정하지 않는다.**

K8s 기본 ConfigMap이나 `.cargo/config.toml`로 containerd의 프록시·이미지 Pull 인증 문제를 해결할 수 없다. 이는 Registry/노드와 ServiceAccount/ImagePullSecret 측에서 관리한다.

## D. Upstream references

- [Buzz chart values](https://github.com/block/buzz/blob/3debf18f1448334aa7bc202d328d93bb9651e168/deploy/charts/buzz/values.yaml)
- [Buzz chart lock](https://github.com/block/buzz/blob/3debf18f1448334aa7bc202d328d93bb9651e168/deploy/charts/buzz/Chart.lock)
- [CloudPirates postgres pinned values](https://github.com/CloudPirates-io/helm-charts/blob/postgres-0.19.5/charts/postgres/values.yaml)
- [CloudPirates redis pinned values](https://github.com/CloudPirates-io/helm-charts/blob/redis-0.30.3/charts/redis/values.yaml)
