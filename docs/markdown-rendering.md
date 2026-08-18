# Markdown 수식·렌더링 규칙

업데이트: 2026-08-19 KST

## 결론

이 저장소의 display math는 `$$` delimiter 대신 GitHub가 공식 지원하는 fenced `math` 블록으로 작성한다. 두 문법 모두 GitHub 웹에서 지원되지만, fenced 문법은 문단·목록·표 경계와의 충돌을 줄이고 원문에서도 수식 범위를 명확히 보여준다.

GitHub 공식 문서가 명시적으로 보장하는 범위는 GitHub 웹의 Issues, Discussions, Pull Requests, Wiki, Markdown 파일이다. GitHub Mobile 앱의 수식 렌더링 지원은 같은 문서에서 보장하지 않는다. 따라서 다음처럼 판정한다.

- GitHub 웹에서 렌더링되고 정적 검증을 통과함: 저장소 소스 정상
- 앱에서 delimiter나 LaTeX 원문이 보임: 앱 클라이언트 렌더러 문제일 가능성이 큼
- 웹에서도 원문이 보임: source delimiter, Markdown 경계, LaTeX 문법을 다시 검사

## 작성 규칙

### Display math

````markdown
```math
\operatorname{TTFT} \approx T_{queue} + T_{prefill} + T_{route}
```
````

- fence 앞뒤에 빈 줄을 둔다.
- fence 안에는 `$$`를 다시 넣지 않는다.
- 여러 줄 정렬은 `\begin{aligned} ... \end{aligned}`처럼 수식 환경 안에서 처리한다.
- 표 셀 안에는 display math를 넣지 않는다. 짧은 inline math 또는 표 아래 별도 블록을 쓴다.

### Inline math

짧은 식은 `$O(T^2)$`, `$d_{head}$`처럼 한 줄 안에서 닫는다. Markdown 문법과 겹치는 문자가 많은 경우 GitHub의 backtick-delimited inline math 문법을 고려한다.

### 코드와 금액 기호

- shell 변수, 명령, 가격처럼 수식이 아닌 `$`는 inline code 또는 fenced code에 둔다.
- 수식 내부의 실제 달러 기호는 `\$`로 escape한다.

## 전체 검증

저장소 루트에서 실행한다.

```bash
node tools/validate_markdown.mjs .
```

검사 범위:

- 모든 Markdown fence의 열림/닫힘
- raw `$$` 잔존 여부
- inline `$` delimiter 짝
- display-math block의 brace와 environment 균형
- 상대 링크의 실제 대상
- 빈 Mermaid block

이 검사는 MathJax 전체를 재구현하지 않는다. 최종 확인은 GitHub 웹의 렌더링 결과로 하고, 앱 표시 차이는 별도 client compatibility 이슈로 취급한다.

## 2026-08-19 일괄 정리

- 감사 도중 main에 추가된 Kimi-K3 모델 문서까지 포함해 display-math 443개를 fenced `math`로 변환했다.
- 변환 전 모든 `$$`는 독립된 줄에 있었고 block delimiter의 짝은 맞았다.
- 변환 후 raw `$$`, 깨진 fence, 불균형 brace/environment, 누락된 상대 링크가 없는지 전체 검사한다.
