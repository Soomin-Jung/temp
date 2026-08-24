# Offline RHEL NVIDIA GPU Maintenance Context

업데이트: 2026-08-24 KST  
상태: Operational reference

## 환경

- RHEL 8.10 계열 오프라인 GPU 서버
- NVIDIA local-repo RPM 사용
- Kubernetes 계열 runtime으로 k3s 사용
- GPU workload는 vLLM 등 inference serving
- H100/L40S non-NVSwitch 시스템과 H200 NVSwitch 시스템이 공존

## H200 NVSwitch 계열 업그레이드 원칙

H200 NVSwitch 시스템은 NVIDIA driver만 올리는 것으로 끝나지 않는다. **driver와 Fabric Manager 버전/stream 정합성**을 같이 관리한다.

대표 절차:

1. 대상 local repository / module stream 확인
2. NVIDIA driver와 Fabric Manager를 같은 기준으로 설치 또는 업그레이드
3. GPU workload 중지
4. Fabric Manager 중지
5. reboot
6. Fabric 상태가 정상 완료 상태인지 검증
7. GPU/NVLink/NVSwitch topology 및 CUDA/NCCL 기본 검증
8. k3s / GPU workload 복구

RHEL module 기반 환경에서는 예를 들어 driver stream과 일치하는 Fabric Manager package/module을 선택해야 한다.

## 운영 원칙

- driver/FM mismatch 상태에서 workload를 먼저 올리지 않는다.
- H100/L40S non-NVSwitch node와 H200 NVSwitch node의 maintenance runbook을 동일하게 취급하지 않는다.
- 오프라인 환경이므로 package/repository 반입본과 설치 버전을 사전에 고정한다.
- maintenance 완료 판정은 package install 성공이 아니라 Fabric/GPU/runtime validation까지 포함한다.

이 문서는 과거 NVIDIA/Fabric Manager 업그레이드 대화의 canonical 요약으로 사용한다.
