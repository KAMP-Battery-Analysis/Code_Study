# KAMP - AI 프로젝트 진행 방향

- 회의 일자 : 2024.05.07

### 1. 페르소나
- 배터리 연구/개발 관련 시험자

- 배터리 제조 품질/기술 엔지니어
  - 누구나 시험은 할 수 있고, 합/불 판정은 할 수 있으나 경계선 데이터의 경우 숙련자의 판단이 필요하다
  - 미숙련자에게 판단을 도울 수 있는 툴(시스템)을 제공하기 위함
  - 이상 유무 판단에 대한 최적화된 기준 필요
    - 세부데이터 : 모형의 성능, 불량 원인

---

### 2. 사전 공부 필요 사항
- MTadGAN, PCA 알고리즘 분석 및 이해
- KAMP 데이터셋 코드 구조 이해 Study (필수 기반 사항)

### 2. 기본적으로 구현할 기능 제안
1. 모델의 성능을 개선(보완)

- Parameter 조정을 통한 모델 최적화

- LSTM -> Transformer 알고리즘 변경

2. 모델 개선 산출물로 성능 및 성능 개선 분석 보고서
- 모니터링 시스템 구축

  - (장비 데이터 추출 -> csv 파일 변환) -> 데이터 전처리 -> AI 분석 -> 결과 피드백

  - 실시간 데이터 업데이트 및 결과 예측(현 시점 이상 유무)
    - 실시간 인터페이스 출력