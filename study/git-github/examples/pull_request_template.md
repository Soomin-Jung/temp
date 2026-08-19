## 문제와 목적

<!-- 현재 문제, 이 변경이 필요한 이유, 관련 issue/incident를 적습니다. -->

## 변경 내용

<!-- 주요 변경과 의도적으로 바꾸지 않은 범위를 적습니다. -->

## 영향

- 사용자/API 호환성:
- 성능·용량:
- 보안·권한:
- 운영·관측:

## 검증

<!-- 실행한 정확한 command, test case, 결과를 적습니다. -->

```text
command:
result:
```

## 배포와 롤백

- migration 순서:
- feature flag 또는 단계적 rollout:
- rollback/roll-forward:
- 확인할 metric/log:

## 리뷰 포인트

<!-- reviewer가 특별히 집중할 설계나 위험을 적습니다. -->

## Checklist

- [ ] 변경 범위 밖의 파일이 포함되지 않았습니다.
- [ ] 새 동작과 실패 경로를 검증했습니다.
- [ ] 문서·schema·generated output을 필요한 만큼 갱신했습니다.
- [ ] secret, 내부 주소, 불필요한 artifact를 포함하지 않았습니다.
- [ ] backward compatibility, migration, rollback을 검토했습니다.
