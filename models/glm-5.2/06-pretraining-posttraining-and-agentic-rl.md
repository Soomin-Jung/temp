# 06. Pretraining, Post-training & Agentic RL

작성일: 2026-08-25  
핵심 원문: GLM-5 Technical Report + GLM-5.2 official blog

## 1. Agent capability를 architecture만으로 설명하면 틀린다

GLM-5.2가 coding/agent benchmark에서 강한 이유를 `1M context`, `DSA`, `MoE`만으로 설명하면 절반만 본 것이다.

모델 capability는 최소 다음 네 층으로 분리한다.

```text
Architecture
  MLA / DSA / IndexShare / MoE / MTP

Pre/Mid-training
  massive corpus / context curriculum / sparse adaptation

Post-training
  reasoning RL / agentic RL / general RL / OPD

Agent system
  sandbox / tools / rollout orchestration / verifier / critic
```

architecture는 **가능한 computation graph**를 정하고, training은 그 graph가 실제로 어떤 behavior를 학습하는지를 정한다.

---

## 2. GLM-5의 base-training foundation

GLM-5 technical report는 총 약 28.5T-token 규모의 training을 설명한다.

큰 흐름은:

```text
Base pretraining
  large heterogeneous corpus
       │
       ▼
Mid-training
  context extension
  reasoning/coding data
  DSA adaptation
       │
       ▼
Post-training
  reasoning → agentic → general
```

특히 context는 처음부터 200K-class로만 학습하는 것이 아니라 점진적으로 늘린다.

long-context quality는 단순 positional range 확장이 아니라 **길어진 context에서 실제로 retrieval/reasoning해야 하는 data distribution**을 함께 학습해야 한다.

---

## 3. DSA transition 자체도 training problem이다

DSA는 top-k token 외 history를 attention에서 제거한다.

잘못 고르면 정보가 사라진다.

따라서 indexer가 dense attention의 중요한 token을 모방하도록 warm-up한 뒤 sparse mode에 적응하는 continued-pretraining phase가 필요하다.

```text
Dense teacher attention
      │
      ▼ distillation / warmup
DSA indexer
      │
      ▼
Sparse attention adaptation
```

GLM-5 report는:

- indexer warm-up
- sparse adaptation
- 약 20B-token additional training

을 기술한다.

GLM-5.2 IndexShare도 같은 철학을 따른다. 128K mid-training부터 shared index pattern을 포함해 학습한다.

즉 sparse architecture는 **kernel을 바꾸는 것이 아니라 모델이 무엇을 버릴지 다시 배우는 과정**이다.

---

## 4. GLM-5 post-training sequence

GLM-5 report는 post-training을 하나의 RL stage로 합치지 않고 sequence로 설명한다.

```text
Reasoning RL
    ↓
Agentic RL
    ↓
General RL
    ↓
On-policy cross-stage distillation
```

이 순서의 의미는 domain capability를 순차적으로 강화하면서 뒤 stage가 앞 stage capability를 파괴하지 않도록 policy distribution을 다시 정렬하는 데 있다.

### Reasoning RL

- math/reasoning
- coding reasoning
- verifiable reward 중심 task

### Agentic RL

- tool use
- coding environment
- multi-turn task
- stateful rollout

### General RL

- broader instruction following
- conversational/general capability

### On-policy distillation

teacher의 static dataset만 학습하는 off-policy distillation과 달리 현재 policy가 실제 생성하는 trajectory를 기준으로 teacher signal을 적용한다.

agent처럼 policy distribution이 빠르게 바뀌는 영역에서 mismatch를 줄이는 장점이 있다.

---

## 5. slime: model training과 rollout engine의 경계

GLM-5는 large-scale post-training framework로 **slime**을 공개·설명한다.

agent RL에서 가장 어려운 부분 중 하나는 GPU에서 policy update를 하는 것보다 **rollout을 대규모로 안정적으로 생성하는 것**이다.

agent rollout은:

```text
prompt
 → model action
 → tool/environment execution
 → result
 → model action
 → ...
 → final outcome
```

처럼 external state를 포함한다.

slime이 해결하려는 영역:

- customizable rollout
- inference/training integration
- asynchronous execution
- long-running environment
- resource scheduling
- fault tolerance

GLM-5 report는 multi-node inference에서 large EP/DP topology, FP8 rollout, MTP, P/D disaggregation 등을 RL infrastructure와 연결한다.

특히 **P/D disaggregation을 production serving뿐 아니라 multi-turn RL rollout interference를 줄이는 수단으로 사용**한다는 점이 흥미롭다.

---

## 6. 왜 asynchronous RL이 중요한가

agent trajectory의 실행시간은 균일하지 않다.

```text
Task A: compile 5 sec
Task B: test suite 2 min
Task C: external tool timeout
Task D: long reasoning 100K tokens
```

synchronous batch training에서는 가장 느린 rollout 때문에 GPU가 기다릴 수 있다.

asynchronous RL은 rollout producer와 training consumer를 분리해 resource idle을 줄인다.

```text
Rollout workers ──┐
Rollout workers ──┼─→ trajectory buffer → trainer
Rollout workers ──┘
```

하지만 async가 깊어지면 policy staleness가 커질 수 있으므로:

- policy version control
- sample freshness
- on-policy correction
- scheduling

이 필요하다.

즉 agent RL infra는 distributed inference + RL systems 문제다.

---

## 7. GLM-5.2: long-horizon task를 별도 target으로 강화

GLM-5.2 official positioning은 단순 coding score 개선보다 **long-horizon task**다.

대표 workload:

- large-scale implementation
- automated research
- performance optimization
- complex debugging
- long coding-agent trajectories

이러한 task는 1M input capability뿐 아니라:

- 수십~수백 turn의 goal 유지
- tool failure 후 recovery
- 과거 constraint recall
- intermediate artifact tracking
- 긴 reasoning/output

이 필요하다.

따라서 1M-context training을 coding-agent scenario로 확장했다는 점이 중요하다.

---

## 8. GLM-5.2의 rollout type 확장

공식 blog는 slime 쪽에서 다음 rollout/workflow를 언급한다.

### White-box rollout

model/internal serving stack을 직접 통제하며 rollout한다.

- logits/hidden/internal state 접근 가능
- training-serving co-design 용이

### Black-box rollout

외부 API 모델/agent와 상호작용하는 trajectory도 다룬다.

### Compact trajectory

긴 interaction history를 그대로 무한 누적하지 않고 training에 유용한 형태로 압축/정리한다.

### Sub-agent workflow

한 parent agent가 sub-agent를 호출하는 구조까지 rollout 대상으로 확장한다.

이 부분은 현대 coding/EDA agent와 매우 직접적으로 연결된다.

---

## 9. Parallel OPD — 여러 expert policy를 하나로 합치기

GLM-5.2는 parallel On-Policy Distillation(OPD)을 이용해 여러 specialized expert model의 capability를 최종 model로 합친다고 설명한다.

개념적으로:

```text
Reasoning expert ─┐
Coding expert ────┤
Agent expert ─────┼─→ current policy trajectories → distill → GLM-5.2
Tool expert ──────┤
...               ┘
```

공식 글은 10개가 넘는 expert model을 병렬 OPD로 합치며 전체 OPD를 약 2일 수준으로 수행했다고 설명한다.

핵심은 offline static teacher outputs를 모아 학습하는 것보다 **현재 student policy가 방문하는 state/action distribution에 teacher knowledge를 적용**한다는 데 있다.

---

## 10. Critic-based PPO와 single-rollout

long-horizon agent에서는 같은 task라도 trajectory가 매우 길고 variable하다.

여러 rollout을 group으로 묶는 방법은 compute cost가 커지고 trajectory compaction 때문에 비교가 복잡해질 수 있다.

GLM-5.2는 long-horizon RL에서 critic-based PPO와 single-rollout setting을 사용한다고 설명한다.

중요한 system implication:

- rollout 하나가 매우 비쌈
- reward signal이 sparse할 수 있음
- value/critic quality가 중요
- tool environment correctness가 reward correctness와 직결

한다.

---

## 11. Agentic RL에서 reward hacking이 특히 위험한 이유

coding/EDA task는 reward가 tool result로 자동화되기 쉬워 보인다.

예:

```text
unit test pass = reward
simulation pass = reward
lint clean = reward
PPA improved = reward
```

그러나 agent가 metric loophole을 찾을 수 있다.

예:

- test를 삭제/무력화
- checker를 수정
- timeout/exception을 pass처럼 처리
- spec 자체를 바꿔 benchmark를 우회
- tool wrapper를 조작

GLM-5.2는 coding-agent RL에서:

- rule-based filtering
- LLM judge
- online tool-call monitoring

등의 anti-hacking을 사용한다고 설명한다.

EDA agent에서는 이 문제가 더 중요하다. RTL이 compile된다고 spec을 만족하는 것이 아니고, PPA가 좋아져도 functional equivalence가 깨질 수 있기 때문이다.

---

## 12. EDA RL로 번역하면 무엇을 배워야 하는가

칩 설계 agent를 RL한다면 reward를 단일 pass/fail로 잡기보다 hierarchy로 설계해야 한다.

```text
functional correctness
  ├─ compile
  ├─ lint
  ├─ simulation
  ├─ formal/equivalence
  └─ assertions

physical quality
  ├─ WNS/TNS
  ├─ area
  ├─ power
  └─ congestion

process validity
  ├─ forbidden file modification 금지
  ├─ testbench tampering 금지
  ├─ tool exit semantics 검증
  └─ reproducibility
```

GLM-5.2의 anti-hacking/long-horizon RL 연구는 chip-agent infrastructure를 설계할 때 foundation model보다 더 직접적인 참고점이 될 수 있다.

---

## 13. Training-serving configuration reuse

GLM-5.2 공식 설명에서 중요한 engineering point 중 하나는 training inference와 production serving config를 최대한 재사용하는 것이다.

왜 중요한가:

```text
RL rollout runtime != production runtime
```

이면:

- parser behavior 차이
- MTP 차이
- quantization 차이
- tool-call format 차이
- max context 차이
- scheduler/cache behavior 차이

때문에 training distribution과 production distribution이 벌어진다.

long-horizon agent에서는 작은 runtime difference가 수십 turn 후 큰 behavior difference로 증폭될 수 있다.

---

## 14. Architecture vs training vs agent harness를 분리해서 평가한다

모델 비교 시 다음을 분리한다.

### Base model competence

동일 prompt / no-agent / 동일 decoding

### Agent competence

동일 harness / 동일 tools / 동일 budgets

### System competence

동일 model / 다른 agent orchestration

### Runtime competence

동일 model+harness / 다른 engine, cache, MTP, context configuration

FluxBench와 ACE-RTL 연구가 보여주는 핵심도 이 분리의 중요성이다.

---

## 15. 한 문장 정리

> **GLM-5.2의 agent 능력은 1M context를 가진 큰 모델이어서 생긴 것이 아니라, sparse long-context architecture를 실제 long-running tool trajectory로 학습시키고 그 rollout infrastructure·reward·anti-hacking까지 함께 설계한 결과다.**

## Sources

- GLM-5 Technical Report: https://arxiv.org/abs/2602.15763
- GLM-5.2 official blog: https://z.ai/blog/glm-5.2
- slime: GLM-5 technical report 및 Z.AI 공개 자료 참조
