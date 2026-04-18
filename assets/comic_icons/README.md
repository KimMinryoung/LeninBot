# Comic Icon Library

Named-object SVG icons for Cyber-Lenin political comics. Designed to be inlined into panel `scene_svg` fragments by the comic-authoring agent.

## Design principles

Per `feedback_comic_visual_vocabulary.md`: **every visual element must be a recognizable, nameable object** (gold bar, TV, vault, document, person) — never abstract rectangles/triangles/dashed circles. The reader must parse the scene in ≤2 seconds without a symbol legend.

Per `feedback_comic_panel_content.md`: panels contain **only image + single speech balloon**. No in-panel captions, headings, subtext, or analysis sections.

## Conventions

- **viewBox**: all icons use `0 0 100 100`. Non-square objects fit within this canvas with whitespace padding.
- **Stroke weight**: 2.5–3 for main outlines; 1–1.5 for detail lines. Consistent across icons.
- **Palette**: bold outlines in `#2d3436`; object-specific semantic colors (gold = `#f4d03f`, red = `#e74c3c`, dollar green = `#9fd8ae`, etc.).
- **Text**: `system-ui, sans-serif` only. No web-font dependencies — comic pages should render without external fetches.
- **No `<script>`, no `<foreignObject>`, no event handlers.**

## How to use in a panel's scene_svg

Each icon file is a standalone `<svg>` element. To embed one into a panel scene, **copy the inner children** and wrap them in `<g transform="translate(x, y) scale(s)">` where `x`/`y` position within the panel viewBox (the composer uses 960×320 per panel) and `s` scales from the 100×100 canvas.

Example: placing a vault at panel coordinate (120, 80) at 1.8× scale:

```xml
<g transform="translate(120, 80) scale(1.8)">
  <!-- contents copied from vault.svg (everything between <svg> and </svg>) -->
</g>
```

You can also mutate the icon (recolor, add a speech line, swap a label) by editing the copied SVG children. These are templates, not immutable assets.

## Current icons (v1)

| File | Object | Typical scene role |
|---|---|---|
| `tv_news.svg` | 뉴스 TV 화면 | 속보·뉴스 헤드라인 장면 |
| `missile_alert.svg` | 미사일 경고 알림 | 전쟁·공포·비상 상황 |
| `chart_up.svg` | 상승 차트 | 가격·지표 상승 |
| `chart_down.svg` | 하락 차트 | 가격·지표 하락 |
| `vault.svg` | 중앙은행 금고 | 국가·중앙은행 보유고 |
| `goldbar_stack.svg` | 금괴 스택 | 금 보유·축적 |
| `dollar_bill.svg` | 달러 지폐 | 달러·기축통화 |
| `sanctions_stamp.svg` | 제재 도장 | 제재·금융무기화 |
| `torn_paper.svg` | 찢어진 약속 문서 | 신뢰 파괴·약속 파기 |
| `speaker_head.svg` | 말하는 사람 | 논평자·시민·인물 |

## Adding new icons

When a new comic topic needs an object not in the library:
1. Draft as a standalone `<svg viewBox="0 0 100 100">` following the conventions above.
2. Add the file here, update the table, commit.
3. Keep each icon under ~30 lines of SVG for readability and LLM-copyability.
