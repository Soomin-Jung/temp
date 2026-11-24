# Dify Enterprise → Community Edition 무손실 마이그레이션 Runbook

업데이트: 2026-11-24 KST  
상태: 운영 적용 전 검증 필요 / PostgreSQL 기반 self-hosted 환경  
범위: Kubernetes, Docker Compose, 폐쇄망, Main DB + Plugin DB + Vector DB + Object Storage

> 공식 EE→CE downgrade 절차 또는 무손실 보증이 아니다. 실제 Enterprise 버전/스키마/계약/구성은 미확인이다. 공개 CE 1.17.1 및 해당 Compose의 Plugin Daemon 0.6.10을 source audit reference로 사용했다. 실제 대상 버전은 별도로 확정한다. 명령은 승인된 격리 환경에서 먼저 검증한다.

## 0. 성공 정의 및 기본 아키텍처

Edition 변경과 버전 업그레이드, DB 엔진 변경, Vector DB 변경, Storage 변경을 분리한다. 첫 목표는 EE 기반 CE revision에 가장 가까운 호환 가능한 CE 환경에서 기존 식별자와 상태를 유지하는 것이다. 보안 패치와 이후 CE 업그레이드는 별도 검증한다.

~~~mermaid
flowchart LR
  EE[EE production] --> B[Consistent backup at T0]
  B --> C[CE isolated candidate]
  BASE[Clean CE baseline] --> D[Schema and data diff]
  C --> D
  D --> V[Integrity and functional tests]
  V --> G{Go / No-Go}
  G -->|Go| T[Ingress cutover at T1]
  G -->|No-Go| R[Preserve EE / remediate CE]
~~~

- RPO=0 목표: 최종 쓰기 차단 시각 T0 이전에 커밋된 이관 대상 데이터 모두 복원.
- RTO: 리허설에서 백업·전송·복원·검증·라우팅 시간 실측 후 승인. 대용량 DB/Vector DB의 downtime을 임의 추정하지 않는다.
- 데이터 보존과 기능 보존은 별개다. CE에서 사용 불가능한 EE Audit/RBAC/SSO/조직/환경 관련 데이터는 암호화된 원본 및 승인된 조회용 archive로 보존한다. 기능 대체/손실 허용은 데이터 소유자의 명시적 승인이 필요하다.
- tenant/app/account/dataset/document/chunk/conversation ID, 기존 API Key 및 published URL, plugin identity, credential을 가능한 한 유지한다.
- 진행 중인 실행 및 외부 도구 side effect, pause/resume, trigger는 DB dump만으로 exactly-once 복구되지 않는다. 완료/중단/재처리 목록을 관리한다.
- Go 조건: 미승인 유실 0, 신규 참조 불일치 0, 중요 기능 실패 0, 권한 확대 0, 미검증 migration revision 0.

## 1. 근거 수준 및 라이선스 Gate

| 구분 | 근거 | 판단 |
|---|---|---|
| 확인 | CE 1.17.1 LICENSE | 다중 workspace 환경은 Dify의 서면 승인 없는 운영이 제한된다. 최종 사용자 수와 workspace 수는 별개다. |
| 확인 | CE 소스/Compose | DB, Plugin, 파일, Vector Store, Agent 상태 분리 |
| 확인 | EE release portal | 릴리스별 Community 기반 버전/commit 정보 확인 가능 |
| 미확인 | 운영 EE schema/patch | 기반 CE commit과 동일한 DB schema라고 단정할 수 없다 |
| 미확인 | 계약 및 만료 동작 | grace period, API/관리자 접근, export, rollback 권리를 공급사와 확인 |
| 권고 | 본 Runbook | 별도 CE 복원 후 전환. 공개 자료만으로 EE→CE 완전 호환성을 보증할 수 없다 |

공급사 확인 사항: 공식 export/downgrade 도구, 정확한 기반 CE commit과 migration revision, EE 전용 DB/table/column, EE 전용 key provider 호환성, plugin runtime 이관 경로, 만료 후 export/rollback 허용 기간, 다중 workspace 사용 범위. 계약/EE 비공개 자료는 공개 저장소에 넣지 않는다.

다중 workspace이면 법률 및 보안 승인 전 통합/전환하지 않는다. tenant_id 일괄 UPDATE, workspace UI 숨기기, license check 우회는 해결책이 아니다. 인스턴스 분리 또한 라이선스 적용 범위를 서면 확인한다.

## 2. Source-level audit (CE 1.17.1 reference)

| 위치 | 확인한 구현 | 마이그레이션 영향 |
|---|---|---|
| [API entrypoint L11–18](https://github.com/langgenius/dify/blob/1.17.1/api/docker/entrypoint.sh#L11-L18) | MIGRATION_ENABLED=true이면 flask upgrade-db, MODE=migration이면 종료 | 동시 시작 대신 단일 migration job |
| [API entrypoint job mode](https://github.com/langgenius/dify/blob/1.17.1/api/docker/entrypoint.sh#L81-L114) | MODE=job Flask CLI 실행 | 동일 이미지의 관리 명령 실행 가능 |
| [Compose](https://github.com/langgenius/dify/blob/1.17.1/docker/docker-compose.yaml) | API storage /app/api/storage, plugin /app/storage, daemon 0.6.10-local, Agent backend 분리 | 실제 mount 및 backend 조사 |
| [env example](https://github.com/langgenius/dify/blob/1.17.1/docker/.env.example) | DB_PLUGIN_DATABASE, PLUGIN_DAEMON_KEY, PLUGIN_DIFY_INNER_API_KEY, PLUGIN_STORAGE_LOCAL_ROOT | 별도 DB와 storage, 양측 인증 키 일치 |
| [RSA](https://github.com/langgenius/dify/blob/1.17.1/api/libs/rsa.py) | local provider의 privkeys/{tenant_id}/private.pem, hybrid RSA/AES 복호화 | tenant UUID, key 보존 |
| [encrypter](https://github.com/langgenius/dify/blob/1.17.1/api/core/helper/encrypter.py) | key_provider_manager에 위임 | 외부 key provider 별도 검증 |
| [account model](https://github.com/langgenius/dify/blob/1.17.1/api/models/account.py) | accounts.password nullable, tenants.encrypt_public_key, tenant_account_joins | SSO 계정의 로컬 로그인 보장 없음 |
| [provider model](https://github.com/langgenius/dify/blob/1.17.1/api/models/provider.py) | providers, provider_models, provider_credentials, provider_model_credentials, load_balancing_model_configs | credential뿐 아니라 provider/LB 설정 복구 |
| [app/data model](https://github.com/langgenius/dify/blob/1.17.1/api/models/model.py) | apps, sites, api_tokens, end_users, conversations, messages, message_files, upload_files.key/size | URL/API key/history/파일 연결 검증 |
| [workflow model](https://github.com/langgenius/dify/blob/1.17.1/api/models/workflow.py) | workflows, workflow_runs, workflow_node_executions, offload/archive, pause, draft/conversation variables, secret variable 암복호화 | 그래프 외 실행 이력 및 파일, resume 상태 보존 |
| [dataset model](https://github.com/langgenius/dify/blob/1.17.1/api/models/dataset.py) | dataset_collection_bindings, document_segments.index_node_id, child_chunks, metadata, external_knowledge_bindings, pipelines | SQL metadata와 vector mapping 검증 |
| [Plugin Daemon 0.6.10 DB init](https://github.com/langgenius/dify-plugin-daemon/blob/0.6.10/internal/db/init.go) | GORM AutoMigrate 대상은 plugin/installations/declarations/endpoints/runtime/tenant storage 등, 기존 declaration column의 NOT NULL 해제 DDL 포함 | API MIGRATION_ENABLED=false가 plugin DDL을 차단하지 않는다. 해당 버전 호출 경로 검증 및 격리 기동 |

이 목록은 CE reference이며 운영 EE schema inventory가 아니다. 실제 버전별 소스, catalog, data migration을 대조한다.

## 3. Phase A — Discovery

### 3.1 변경 티켓에 기록할 필드

| 항목 | 값 |
|---|---|
| EE chart/compose 및 전체 image tag/digest | TBD |
| 라이선스 만료 시각(시간대), 유예 및 rollback 권리 | TBD |
| 기반 CE tag/commit, image digest, migration heads | TBD |
| DB 엔진/version/locale/extensions, Main/Plugin/EE/기타 DB | TBD |
| Storage endpoint/bucket/prefix/PVC/암호화 | TBD |
| Vector DB version/topology/collection/index/shard | TBD |
| Redis DB index/prefix/broker/backend/persistence | TBD |
| Workspace/account/app/dataset/document/run/message/plugin 수량 | TBD |
| SSO/MFA/RBAC/환경 분리, API/webapp 공개 정책 | TBD |
| 외부 webhook, trigger, cron, SMTP, tool, model, egress | TBD |
| Agent workspace/snapshot 및 plugin sandbox 사용 여부 | TBD |
| 담당자, 승인자, 데이터 소유자, 보안 담당 | TBD |

비밀값 대신 secret reference 또는 보호된 내부 저장소 위치만 기록한다.

Kubernetes inventory:

~~~bash
set -euo pipefail
umask 077
NS='<source-namespace>'
REL='<helm-release>'
B="./dify-migration-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$B"/{inventory,db,storage,vector,validation}
kubectl config current-context
kubectl -n "$NS" get deploy,sts,ds,job,cronjob,hpa,svc,ingress,pvc,networkpolicy -o wide \
  > "$B/inventory/k8s.txt"
kubectl -n "$NS" get pods -o json | jq -r '
  .items[] | .metadata.name as $p |
  (.status.initContainerStatuses[]?, .status.containerStatuses[]?) |
  [$p,.name,.image,.imageID] | @tsv' > "$B/inventory/images.tsv"
helm list -n "$NS" > "$B/inventory/helm.txt"
helm get values "$REL" -n "$NS" --all > "$B/inventory/helm-values.yaml"
helm get manifest "$REL" -n "$NS" > "$B/inventory/helm-manifest.yaml"
kubectl -n "$NS" get deploy,sts,ds,job,cronjob,cm,secret,svc,ingress,pvc,hpa,networkpolicy -o yaml \
  > "$B/inventory/resources.yaml"
~~~

별도 PV/CSI VolumeSnapshotClass, ExternalSecret/SecretStore, cert-manager, Operator CR, GitOps/autoscaling 설정 및 외부 저장소도 확인한다. 복구 시 기존 manifest의 UID/resourceVersion/status를 그대로 적용하지 않는다.

Compose inventory:

~~~bash
docker compose config --services
docker compose config --images
docker compose ps -a
docker compose config > "$B/inventory/compose-rendered.yaml"
~~~

.env, service별 env, override YAML, proxy 설정, volume mounts, secret files 별도 보존. 출력은 비밀값을 포함하므로 공개 로그/Git에 올리지 않는다.

### 3.2 PostgreSQL read-only 조사

PGSERVICE와 권한 0600의 pgpass 또는 승인된 Secret 주입으로 DB 연결을 관리한다. 다음 SQL은 PostgreSQL 기준이며 MySQL이면 엔진별 백업 및 schema diff로 대체한다.

~~~sql
SELECT current_database(), current_user, version();
SELECT datname, pg_encoding_to_char(encoding) AS encoding, datcollate, datctype,
       pg_size_pretty(pg_database_size(datname)) AS size
FROM pg_database WHERE NOT datistemplate ORDER BY datname;
SELECT extname, extversion FROM pg_extension ORDER BY extname;
SELECT table_schema,table_name FROM information_schema.tables
WHERE table_type='BASE TABLE'
  AND table_schema NOT IN ('pg_catalog','information_schema')
ORDER BY 1,2;
SELECT * FROM alembic_version;
SELECT id,name,status FROM tenants ORDER BY id;
SELECT tenant_id,role,count(*) FROM tenant_account_joins GROUP BY 1,2 ORDER BY 1,2;
SELECT tenant_id,count(*) FROM apps GROUP BY 1 ORDER BY 1;
SELECT tenant_id,count(*) FROM datasets GROUP BY 1 ORDER BY 1;
SELECT count(*) AS accounts, count(*) FILTER
  (WHERE password IS NULL OR password='') AS no_local_password FROM accounts;
~~~

Main DB 외 Plugin/EE DB의 schema-only dump 및 table inventory도 확보한다. Alembic은 해당 테이블이 존재하는 DB에서만 실행. SSO 계정 통계는 password hash를 출력하지 않는다.

### 3.3 버전 매핑 및 폐쇄망 반입

EE release portal의 기반 CE tag/commit 및 component matrix 확인 → CE release note, migrations/versions, docker 구성, plugin daemon tag 확인 → EE patch commit과 CE tag 차이 분석. 동일 기반 버전도 스키마 호환 보장은 아니다.

이미지 digest, plugin package와 서명 검증 정보, Python wheelhouse/사내 package index, 시스템 라이브러리, sandbox dependencies, CA, chart/manifests를 반입한다. EE 비공개 소스 및 계약은 내부에서만 취급한다.

## 4. Phase B — 전체 백업 및 일관성 확보

### 4.1 백업 대상

| 상태 저장소 | 데이터 |
|---|---|
| Main PostgreSQL | app/workflow/draft/history, account/tenant, API key, dataset metadata, credential |
| Plugin PostgreSQL | installation, declaration, endpoint, runtime, tenant storage |
| EE 전용 DB 및 기타 DB | audit, org, identity, 정책 및 실제 추가 state |
| File/Object Storage | upload, generated files, archive/offload, attachments, privkeys |
| Plugin Storage | packages/assets/persistent data/runtime |
| Vector DB | vectors/index/payload/collection/shard configuration |
| Redis | queue, cache, session, locks, Agent state 등 |
| Agent/sandbox | workspace/snapshot, 실행 dependency 및 설정 |
| Config/Secret | 암호화·인증 key, endpoint, TLS, deployment config |

pg_dump는 개별 DB 안의 일관된 snapshot이지 DB/Plugin DB/Vector DB/Object Storage 간 분산 transaction이 아니다. 사전 리허설 snapshot은 running 상태에서 확보할 수 있으나 논리적으로 어긋날 수 있음을 기록한다. 최종본은 모든 writer가 정지된 공통 무쓰기 구간에서 확보한다.

최종 T0: 신규 ingress/API/webhook 차단 → trigger/beat/CronJob 중지 → active/reserved/scheduled 작업 및 indexing drain → API/Worker/Plugin/Agent/EE writer 정지 → HPA/GitOps 재기동 방지 → DB 및 외부 저장소 writer/credential/network fencing 확인. 외부 자동화도 확인한다. 공유 DB의 무관한 서비스까지 중단하지 않는다. application role 또는 기존 DB 세션 차단은 DBA 승인 및 원복 절차 필요. read-only transaction 설정만으로 fencing을 보장할 수 없다.

### 4.2 PostgreSQL backup

PGSERVICE는 예시, pg_dump/pg_restore client는 source/target version과 호환되도록 고정한다. 가능한 동일 major, encoding/collation, extension 사용. 관리형 DB는 native snapshot/PITR을 보조 확보한다.

~~~bash
set -euo pipefail
umask 077
export PGSERVICE=ee-backup
# 실제 DB 이름 전체로 수정; EE 전용 DB도 포함
for db in dify dify_plugin; do
  pg_dump -Fc --no-owner --no-acl --file="$B/db/$db.dump" "$db" \
    2>"$B/db/$db.dump.log"
  pg_dump --schema-only --no-owner --no-acl "$db" > "$B/db/$db.schema.sql"
  pg_restore --list "$B/db/$db.dump" > "$B/db/$db.toc"
done
(cd "$B/db" && sha256sum ./*.dump > SHA256SUMS && sha256sum -c SHA256SUMS)
# 필요시 DBA가 pg_dumpall --globals-only로 role/global 객체 별도 보관
~~~

--no-owner/--no-acl은 target role로 소유권 재설계 목적이다. 원본 grant/default privileges/role membership/RLS를 별도 보존하고 target 최소 권한으로 재적용한다. dump에는 실행 가능한 DDL이 포함될 수 있으므로 신뢰 및 검토 필요. TOC와 checksum 검증은 실제 restore 시험을 대체하지 않는다.

### 4.3 File/Object Storage

upload_files만 복사하면 workflow offload/archive, 생성 파일, Agent snapshot/workspace, RSA private key 등이 누락될 수 있다. 실제 prefix 및 mount 전체 inventory 필요.

PVC: 앱 정지 후 CSI snapshot 및 신규 restore PVC 또는 read-only backup pod로 tar. VolumeSnapshot.readyToUse와 restore 성공 확인. 실행 중인 PG/vector data directory를 tar하지 않는다.

~~~bash
tar --numeric-owner -czpf "$B/storage/app.tar.gz" -C "$APP_STORAGE_MOUNT" .
tar --numeric-owner -czpf "$B/storage/plugin.tar.gz" -C "$PLUGIN_STORAGE_MOUNT" .
tar -tzf "$B/storage/app.tar.gz" > "$B/storage/app-files.txt"
tar -tzf "$B/storage/plugin.tar.gz" > "$B/storage/plugin-files.txt"
~~~

빈 target에 복구하고 archive path/symlink 안전성, uid/gid, ACL/xattr, FSGroup, mount permission 확인. venv compiled extension은 CPU architecture/ABI가 같을 때만 재사용; 불일치 시 package/persistent data 유지하고 runtime만 재구축.

S3/MinIO 예시(별도 승인된 alias):

~~~bash
mc mirror --preserve source/dify-app target/dify-app
mc mirror --preserve source/dify-plugin target/dify-plugin
mc ls --recursive --json source/dify-app > "$B/storage/source-app.jsonl"
mc ls --recursive --json target/dify-app > "$B/storage/target-app.jsonl"
~~~

mirror가 과거 version/delete marker/IAM/KMS 정책까지 자동 복원하는 것은 아니다. versioning, object lock, retention, SSE-KMS 및 metadata를 별도 보존. 사전 copy + 최종 delta 시 삭제된 object까지 manifest와 비교. 운영/공유 bucket에 --remove 사용 금지. multipart ETag를 SHA-256으로 간주하지 않는다. key/size/checksum/missing/unexpected key 확인.

### 4.4 Credential 암호화

CE local key provider: storage 상대 경로 privkeys/{tenant_id}/private.pem, tenants.encrypt_public_key와 UUID를 모두 보존. workflow secret variable도 tenant key provider를 사용한다. 외부 KMS/HSM 사용 시 동일 key 접근 가능성 및 CE 구현 지원 여부 확인. 미지원이면 공급사 도움을 받아 승인된 복호화/재암호화 설계. 평문 credential을 Git/로그에 남기지 않는다.

SECRET_KEY, OAuth/session/signing key, plugin callback key, Agent 암호화/인증 key, sandbox token, URL 설정 inventory. SECRET_KEY와 tenant RSA private key는 서로 대체 불가. key rotation과 edition migration 분리.

~~~bash
# 실제 private key와 DB public PEM을 보호된 파일에 준비
openssl pkey -in "$KEY_FILE" -pubout -outform DER | sha256sum
openssl pkey -pubin -in "$PUBLIC_KEY_FILE" -outform DER | sha256sum
~~~

불일치/부재 시 No-Go. 신규 keypair 생성은 기존 ciphertext 복호화 실패를 해결하지 못한다. 각 tenant에서 실제 credential-backed 모델/tool 호출 확인.

### 4.5 Vector DB backup/restore

| Backend | 방식 | 검증 |
|---|---|---|
| Qdrant single-node | collection snapshot 다운로드 후 target upload/recover | 이름/config/exact count/point IDs/payload |
| Qdrant distributed | shard/replica/node별 snapshot 및 공식 분산 복구 | 단일 node snapshot으로 전체 cluster 복구 불가 |
| Milvus | 호환 matrix에 맞는 Milvus Backup | collection/schema/partition/index/load/entity count |
| Weaviate | 활성화된 backup backend, restore SUCCESS 확인 | collection/tenant/UUID/vector config/count |
| pgvector | 해당 PG DB dump 및 extension 호환 | table/index/dimension/operator class |
| Elasticsearch/OpenSearch | repository snapshot 및 restore | index/alias/mapping/shard/document count |

Qdrant 단일 노드 예시. URL/collection은 실제값으로 치환하며 URL encoding 및 해당 버전 API 확인. 인증은 보호된 curl config 또는 승인된 환경변수 사용.

~~~bash
curl -fsS -H "api-key: $SRC_QDRANT_KEY" "$SRC_QDRANT/collections" \
  > "$B/vector/collections.json"
curl -fsS -H "api-key: $SRC_QDRANT_KEY" "$SRC_QDRANT/collections/$COL" \
  > "$B/vector/$COL.config.json"
curl -fsS -X POST -H "api-key: $SRC_QDRANT_KEY" \
  "$SRC_QDRANT/collections/$COL/snapshots?wait=true" > "$B/vector/$COL.snapshot.json"
SNAP=$(jq -er '.result.name' "$B/vector/$COL.snapshot.json")
curl -fsS -H "api-key: $SRC_QDRANT_KEY" \
  "$SRC_QDRANT/collections/$COL/snapshots/$SNAP" -o "$B/vector/$SNAP"
sha256sum "$B/vector/$SNAP"
# target이 source와 분리된 인스턴스인지 별도 확인 필수
curl -fsS -X POST -H "api-key: $DST_QDRANT_KEY" \
  -F "snapshot=@$B/vector/$SNAP" \
  "$DST_QDRANT/collections/$COL/snapshots/upload?priority=snapshot&wait=true"
~~~

모든 데이터셋이 vector 기반은 아니다. keyword-only, child chunks, multimodal, summary index 등 실제 mapping 확인. segment count와 point count를 무조건 1:1로 비교하지 않는다. alias, custom shard key, replication, quantization 등 topology 별도 보존. Re-embedding은 최후 수단이며 model revision/dimension/tokenizer/chunking/metadata를 고정한 승인된 rebuild와 검색 회귀 검증이 필요하다.

### 4.6 Redis/Agent/in-flight

Redis instance/DB별 cache, broker/result, lock, session, agent state, rate limit 분류. native RDB/AOF/managed snapshot 및 config/prefix 보존. cluster는 shard별 범위 확인. queue drain 후 CE는 빈 broker/cache에서 시작하되 지속성이 필요한 상태는 별도 검증 후 복구. EE task/직렬화 payload 무조건 재생 금지. active/reserved/scheduled, broker depth, 중단 workflow/pause와 외부 side effect를 ledger로 기록. Agent Redis 및 snapshot 상태/TTL/재개 가능성 검증. 불가능하면 승인된 재실행 정책. 신규 Redis로 기존 session 만료 가능성을 공지.

## 5. Phase C — 격리된 CE 환경과 복원

세 DB 역할:

- ee_clone: 원본 backup 그대로 복원한 read-only 분석 기준; 애플리케이션 연결 금지.
- ce_baseline: 순정 CE 목표 버전으로 migration/init한 비교 기준.
- ce_candidate: 실제 patch 및 기능 검증을 적용하는 disposable clone.

Main/Plugin/Redis/Vector/Object Storage 모두 격리한다. CE가 production backend에 접근할 수 없도록 IAM 및 network policy를 적용. 외부 tool/SMTP/webhook egress는 mock/allowlist로 제한한다.

~~~bash
set -euo pipefail
export PGSERVICE=ce-admin
# 반드시 격리 target 확인
psql -X -d postgres -c 'SELECT inet_server_addr(),inet_server_port(),current_user,version()'
# target role/locale/encoding/extension은 사전 확정
createdb --template=template0 --owner="$CE_DB_ROLE" ee_clone
createdb --template=template0 --owner="$CE_DB_ROLE" ce_candidate
createdb --template=template0 --owner="$CE_DB_ROLE" ce_plugin_candidate
pg_restore --exit-on-error --single-transaction --no-owner --no-acl \
  --role="$CE_DB_ROLE" -d ee_clone "$B/db/dify.dump"
pg_restore --exit-on-error --single-transaction --no-owner --no-acl \
  --role="$CE_DB_ROLE" -d ce_candidate "$B/db/dify.dump"
pg_restore --exit-on-error --single-transaction --no-owner --no-acl \
  --role="$CE_DB_ROLE" -d ce_plugin_candidate "$B/db/dify_plugin.dump"
~~~

source encoding/collation이 default와 다르면 createdb 옵션에 명시한다. extension 설치 권한은 DBA가 사전 준비. 대규모 DB는 --single-transaction 대신 --jobs=N + --exit-on-error를 리허설(병용 금지). 부분 restore 실패 시 candidate를 폐기하고 빈 DB부터 재실행. 운영 DB에 --clean/--create/DROP 사용 금지.

CE env 예시(대상 버전 실제 chart/compose와 대조):

~~~ini
MIGRATION_ENABLED=false
DEPLOYMENT_EDITION=COMMUNITY
SECRET_KEY=<preserved-secret>
DB_HOST=<candidate-db-host>
DB_DATABASE=ce_candidate
DB_PLUGIN_DATABASE=ce_plugin_candidate
REDIS_HOST=<isolated-redis>
VECTOR_STORE=<same-backend>
STORAGE_TYPE=<same-backend>
PLUGIN_DAEMON_KEY=<matching-secret>
PLUGIN_DIFY_INNER_API_KEY=<matching-callback-secret>
~~~

Console/Service API/WebApp/FILES_URL/INTERNAL_FILES_URL, OAuth redirect/CORS/cookie/TLS CA, SSRF proxy/NO_PROXY, model URL, upload limit 및 timeout도 mapping. 실제 ingress 전환까지 외부 webhook 중복 소비 차단.

## 6. Phase D — Schema compatibility engineering

### 6.1 Schema diff

Main DB와 Plugin DB 각각 EE clone / 순정 CE baseline / candidate의 schema를 비교. 동일 pg_dump client 사용. schema name/type/default/nullability, PK/FK/check/unique/index, trigger/function/view/sequence/enum/extension, RLS, partition, grants 비교. dump header/random 값 등 무관한 차이는 별도 구분한다.

~~~bash
for db in ee_clone ce_baseline ce_candidate; do
  pg_dump --schema-only --no-owner --no-acl "$db" > "$B/db/$db.schema.sql"
done
rc=0
diff -u "$B/db/ee_clone.schema.sql" "$B/db/ce_baseline.schema.sql" \
  > "$B/db/schema.diff" || rc=$?
test "$rc" -le 1
~~~

catalog inventory SQL (각 DB에서 psql -X --csv -v ON_ERROR_STOP=1 -f 실행):

~~~sql
SELECT table_schema,table_name,column_name,ordinal_position,data_type,
       udt_schema,udt_name,is_nullable,column_default,
       character_maximum_length,numeric_precision,numeric_scale
FROM information_schema.columns
WHERE table_schema NOT IN ('pg_catalog','information_schema') ORDER BY 1,2,3;
SELECT n.nspname,c.relname,x.conname,x.contype,pg_get_constraintdef(x.oid),x.convalidated
FROM pg_constraint x JOIN pg_class c ON c.oid=x.conrelid
JOIN pg_namespace n ON n.oid=c.relnamespace ORDER BY 1,2,3;
SELECT schemaname,tablename,indexname,indexdef FROM pg_indexes ORDER BY 1,2,3;
~~~

### 6.2 Migration revision decision tree

~~~bash
# target CE checkout / 실제 image에 포함된 source 대조
git rev-parse HEAD
git ls-files api/migrations
# 격리된 CE 관리 컨테이너에서
flask db current
flask db heads
flask db history
flask --help
~~~

| 상태 | 대응 |
|---|---|
| EE revision이 CE graph에 존재, 실제 schema/data invariant 일치 | 검증된 forward migration |
| EE revision이 CE head와 동일하지만 DDL/data 차이 존재 | edition-specific patch 분석 |
| EE revision이 CE graph에 없음 | 자동 upgrade 중지, revision과 DDL/data backfill 대응 분석 후 bridge/ETL |
| CE 필수 table/column 누락 | 공식 migration 또는 검증된 DDL+backfill |
| EE 추가 NOT NULL/default/trigger/FK가 CE INSERT 차단 | 의미 확인 후 default/constraint/trigger 변경 승인 |
| enum/role 값 불일치 | 의미 mapping 및 권한 검증 |
| CE 미지원 JSON/credential/node type | 공급사 변환/대체 또는 archive |

금지: schema 검증 없이 flask db stamp head, alembic_version 직접 UPDATE, 모든 FK 삭제, session_replication_role=replica로 오류 은폐, EE table 일괄 DROP. stamp는 revision metadata만 변경하므로 모든 DDL/data migration 효과가 실제 적용됐음을 입증한 경우에만 승인 절차로 검토한다. EE 추가 table은 CE에 영향이 없다면 유지 가능하다. 제약 및 보안 정책 검토 필요.

patch는 versioned SQL, precondition, expected row count, postcondition, snapshot restore 방법을 포함. 실제 schema를 확인하기 전에 범용 ALTER script를 작성하지 않는다. CE release note의 별도 data migration도 확인한다.

CE 1.17.1 확인된 entrypoint 방식:

~~~bash
# target candidate만 연결된 단일 CE 관리 컨테이너
MIGRATION_ENABLED=false flask upgrade-db
# 또는 MODE=migration + MIGRATION_ENABLED=true Job
~~~

K8s Job parallelism=1, completions=1, backoffLimit=0, target secretRef와 image digest 사용. 성공 로그, revision, diff, backfill 검증 후 API/Worker 시작. 실패 시 clone 재생성. Plugin Daemon은 별도 DDL 가능하므로 candidate에 단일 인스턴스 기동 후 전후 diff 보존.

### 6.3 ETL fallback

차이가 큰 경우 CE baseline의 explicit column mapping으로 별도 프로젝트. 모든 EE row를 CE 보존 / 변환 보존 / EE archive / 승인된 제외로 분류하고 mapping ledger 유지. 기존 UUID를 유지 가능한 table부터 dependency 순서 이관. FK가 없는 JSON 참조, 암호화 key, plugin 및 파일 경로까지 추적. CE 초기 tenant/account ID 충돌 검증. CE baseline에 무조건 data-only dump 적용 금지.

DSL/app package는 보조 복구 artifact. secret 포함 여부, 의존성, ID 유지 여부는 버전별 차이가 있다. 대화 이력/실제 KB/credential/DB를 대체하지 않는다.

## 7. Phase E — Identity/Plugin/runtime integration

### 7.1 Workspace/인증

단일 workspace의 tenant UUID 유지가 기본. 다중 workspace는 라이선스·보안 승인 전 No-Go. membership/role의 실제 authorization semantics를 UI/API 양쪽 검증. SSO-only password 없는 계정은 기존 account_id 유지하면서 CE 지원 reset/invite 경로를 사용한다. 해당 버전 CLI help 및 계정 선택 방식 검증 후 수행. DB password hash 직접 작성/공통 초기 password 배포 금지.

EE app access policy가 CE에서 사라져 비공개 앱이 익명 공개되지 않는지 negative test 필수. reverse proxy SSO만으로 Dify 내부 session/membership 연동이 완성되지 않는다. 데이터 소유자별 App/Dataset 접근 matrix를 작성해 grant 확대를 검출한다.

### 7.2 Plugin

Plugin DB + package/persistent storage를 함께 복구한다. local/serverless runtime, package identity/version/digest, Python ABI, signature policy, network, callback key 대조. auto-upgrade/cleanup/background task 비활성화 지원 여부 확인. 미지원 EE runtime/plugin은 아카이브 후 대체 승인. 임의 uninstall/reinstall은 installation ID 및 graph 참조 변경 가능성이 있어 기본 해결책이 아니다. daemon 기동 전후 schema/row count 비교.

| 증상 | 우선 확인 |
|---|---|
| credential decrypt error | tenant UUID/private-public key/key provider/storage permission |
| plugin runtime not found | package/installation/runtime mode/ABI/dependency mirror |
| callback 401/403 | daemon SERVER_KEY/client key, inner API key 양측, URL/network |
| workflow graph load error | EE-only node, plugin identifier, JSON schema |
| retrieval empty | dataset→collection, point/payload, embedding dimension, filter/permission |
| file 404/403 | key/prefix, upload_files metadata, signed URL/SECRET_KEY, URL |
| insert fails | EE NOT NULL/default/trigger/FK, grants, revision |
| private app becomes public | 즉시 ingress 차단, auth policy 재검토 |

## 8. Phase F — Acceptance validation

### 8.1 정량 검증

T0 이후 원본 snapshot과 최종 candidate를 비교. 오래된 리허설 snapshot과 비교하지 않는다. smoke가 새 데이터를 생성하므로 T0 검증은 smoke 전에 수행하거나 synthetic IDs 별도 집계.

~~~sql
\set ON_ERROR_STOP on
BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY;
SELECT format('SELECT %L AS table_name, count(*) AS row_count FROM %I.%I;',
              schemaname||'.'||tablename,schemaname,tablename)
FROM pg_tables WHERE schemaname='public' ORDER BY tablename
\gexec
COMMIT;
~~~

대규모 DB는 clone에서 count. count가 같아도 ID가 바뀔 수 있으므로 정렬 PK hash 비교. 다음은 id가 존재하는 reference schema 예시. 복합 PK는 전체 tuple 사용.

~~~bash
set -euo pipefail
TABLES='accounts tenants apps workflows datasets documents document_segments conversations messages workflow_runs upload_files api_tokens'
for db in ee_clone ce_candidate; do
  for table in $TABLES; do
    psql -X -v ON_ERROR_STOP=1 -At -d "$db" \
      -c "SELECT id::text FROM public.$table ORDER BY id::text COLLATE \"C\"" \
      | sha256sum > "$B/validation/$db.$table.pk.sha256"
  done
done
for table in $TABLES; do
  diff -u "$B/validation/ee_clone.$table.pk.sha256" \
          "$B/validation/ce_candidate.$table.pk.sha256"
done
~~~

table 목록은 실제 inventory 기반 수정. 누락/실패 숨기지 않는다. 중요 컬럼의 내용 hash도 동일 projection/timezone/encoding/ordering으로 streaming 산출. 변환 대상은 승인된 mapping과 비교. raw PK/credential 보호.

### 8.2 참조 무결성

~~~sql
SELECT count(*) AS app_orphans FROM apps a
LEFT JOIN tenants t ON a.tenant_id=t.id WHERE t.id IS NULL;
SELECT count(*) AS document_orphans FROM documents d
LEFT JOIN datasets ds ON d.dataset_id=ds.id WHERE ds.id IS NULL;
SELECT count(*) AS segment_orphans FROM document_segments s
LEFT JOIN documents d ON s.document_id=d.id WHERE d.id IS NULL;
SELECT count(*) AS cross_dataset_segments FROM document_segments s
JOIN documents d ON s.document_id=d.id WHERE s.dataset_id<>d.dataset_id;
SELECT count(*) AS message_orphans FROM messages m
LEFT JOIN conversations c ON m.conversation_id=c.id WHERE c.id IS NULL;
SELECT count(*) AS membership_orphans FROM tenant_account_joins j
LEFT JOIN accounts a ON j.account_id=a.id
LEFT JOIN tenants t ON j.tenant_id=t.id WHERE a.id IS NULL OR t.id IS NULL;
~~~

원본에 이미 있는 orphan과 신규 발생을 분리. graph JSON의 dataset/app/tool/plugin ID, archived/offloaded output, file/Agent snapshot references는 별도 parser 및 object manifest로 검증. upload_files storage_type/key/size와 실제 object 대조. remote file access/URL 만료 정책 검증. vector collection별 exact count/point ID/payload 및 retrieval 테스트.

### 8.3 기능/보안 Acceptance

| 영역 | 필수 검증 |
|---|---|
| 인증 | 관리자, 일반, SSO-only 계정 대체 인증. 비로그인/권한 없는 계정 거부 |
| API | 기존 key로 chat/workflow blocking/streaming, 잘못된 key 거부 |
| 대화 | 동일 user identity 및 기존 conversation_id로 후속 대화와 history |
| App | 기존 domain/path/code, publish/draft, webapp access policy |
| Workflow | LLM, conditional/loop, tool, HTTP request, code/sandbox, secret env, 기존 run/node/archive history |
| Dataset | retrieval/rerank/metadata filtering/child chunk/external KB, 신규 indexing/수정/삭제 |
| Plugin | provider/custom plugin, install/import/OAuth callback, persistence/restart |
| Storage | 기존 첨부/신규 업로드/서명 URL/archive output/Agent workspace |
| Trigger | mock webhook/schedule, 중복 실행 없음 |
| 운영 | queue depth, DB latency, error rate, pod restart, readiness, throughput/p95 |

~~~bash
curl -fsS -X POST "$CE_API/v1/workflows/run" \
  -H "Authorization: Bearer $TEST_APP_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"inputs":{},"response_mode":"blocking","user":"migration-smoke"}'
~~~

실제 app input schema에 맞춰 fixture 작성. 출력은 보호된 환경에서 확인. 모델 출력은 stochastic이므로 문자열 동일성보다 API contract/참조/도구 호출 성공 및 검색 품질 검증. 외부 side effect tool은 mock 사용.

## 9. Phase G — Cutover / rollback

### 9.1 리허설

동일 규모 clone으로 end-to-end restore 및 rollback 최소 1회 수행. 단계별 백업/전송/restore/patch/vector/validation/라우팅 시간, 임시 디스크 peak 기록. 라이선스 유효 기간 안에 EE rollback 가능 확인. 실패 조건과 담당자 명시. 원본 변경 금지.

### 9.2 운영 변경

| 순서 | 작업 | Gate |
|---|---|---|
| 1 | 변경 승인, 공지, trigger/외부 신규 유입 차단 | 승인 티켓/maintenance response |
| 2 | 진행 workflow/indexing/queue drain | task ledger, pending 0 또는 승인 예외 |
| 3 | 모든 EE writer 및 재기동 차단, storage fencing | writer inventory, T0 UTC |
| 4 | 최종 DB/Plugin/EE/Vector/Object/Redis/Agent backup | 동일 무쓰기 구간, hash/manifest |
| 5 | 빈 CE target restore 및 bridge patch | exit 0/revision/diff |
| 6 | CE plugin/API/worker 단계적 시작(외부 trigger 차단 유지) | readiness 및 credential |
| 7 | T0 데이터 비교, 내부 smoke, ACL negative test | 미승인 차이 0 / P0 실패 0 |
| 8 | 기존 domain routing CE 전환 | TLS/API/WebApp/LB, EE 차단 |
| 9 | 사용자 쓰기 허용 및 trigger 단계적 활성화 | T1 시각, owner 승인 |
| 10 | 모니터링/구 환경 보존 | error rate/queue/backup/retention |

K8s writer 중단 예시 (실제 대상 이름과 HPA/GitOps/CronJob 상태 먼저 확인):

~~~bash
kubectl -n "$EE_NS" scale deployment "$EE_API" "$EE_WORKER" --replicas=0
kubectl -n "$EE_NS" get pod,job,cronjob,hpa
~~~

두 deployment만 정지하는 것으로 fencing이 완료되지 않는다. 별도 beat/trigger/enterprise/plugin/agent/job/external scheduler/operator/기존 streaming connection도 확인. EE와 CE가 backing store를 공유하거나 동시에 writer가 되지 않도록 한다. DNS TTL 변경만으로 keep-alive가 차단되지 않는다.

### 9.3 Rollback

T1 이전: CE writer 및 routing 차단 → synthetic test 상태 폐기 → EE license/DB 상태 확인 → EE writer/ingress 재개. 원본 EE 데이터는 변경되지 않아야 한다.

T1 이후: 단순 LB 복귀 금지. CE 신규/변경/삭제 row, 파일/vector/plugin 상태, 외부 side effect를 보존하고 역이관 호환성과 재실행 안전성을 검증한다. 역이관 경로 미검증이면 CE forward-fix 또는 추가 점검 모드가 기본.

EE license 만료로 재기동 불가하면 원본 DB 보존만으로 서비스 rollback은 불가능할 수 있다. CE 자체 snapshot과 복구 가능한 baseline, 공급사 임시 연장/추출 지원을 준비. 전환 직후 CE 첫 일관성 백업 확보. EE 삭제는 안정화 및 보존 기간 종료와 보안/데이터 소유자 승인 후 별도 변경.

## 10. 변경 승인 양식

| 항목 | 결과 |
|---|---|
| EE/CE 버전·digest·schema revision | TBD |
| Workspace/계약/권한 승인 | TBD |
| 백업 ID 및 검증 hash 위치(보안 경로) | TBD |
| 리허설 RTO 실측 및 목표 downtime | TBD |
| T0/T1, writer fencing 증거 | TBD |
| table/PK/object/vector 정량 검증 | TBD |
| EE-only archive/변환/중단 작업 예외 | TBD |
| 기능/권한 negative test | TBD |
| rollback pre/post T1 검증 | TBD |
| Go/No-Go 승인자 및 시각 | TBD |

## 11. 공식 참고 자료

- [Dify CE 1.17.1 source](https://github.com/langgenius/dify/tree/1.17.1)
- [Dify License](https://github.com/langgenius/dify/blob/1.17.1/LICENSE)
- [Enterprise release portal](https://ee.dify.ai/) — 실제 기반 CE commit 확인. 계약/비공개 정보는 내부 보관.
- [Enterprise features](https://dify.ai/pricing/dify-enterprise)
- [CE Release Notes](https://github.com/langgenius/dify/releases), [migration source](https://github.com/langgenius/dify/tree/1.17.1/api/migrations)
- [Plugin Daemon 0.6.10](https://github.com/langgenius/dify-plugin-daemon/tree/0.6.10)
- [PostgreSQL pg_dump](https://www.postgresql.org/docs/current/app-pgdump.html), [pg_restore](https://www.postgresql.org/docs/current/app-pgrestore.html), [SQL dump backup](https://www.postgresql.org/docs/current/backup-dump.html)
- [Qdrant snapshot](https://qdrant.tech/documentation/operations/snapshots/), [distributed tutorial](https://qdrant.tech/documentation/database-tutorials/create-snapshot/), [upload recovery](https://api.qdrant.tech/api-reference/snapshots/recover-from-uploaded-snapshot)
- [Milvus Backup](https://milvus.io/docs/milvus_backup_overview.md)
- [Weaviate backup and restore](https://docs.weaviate.io/deploy/configuration/backups)

실제 EE schema/구성/계약을 확인한 뒤에야 환경 전용 migration SQL, identity mapping 및 실행 자동화를 확정할 수 있다. 본 문서는 그 실측을 위한 운영 Runbook이며 미검증 환경에 바로 적용하는 일괄 변환 스크립트가 아니다.
