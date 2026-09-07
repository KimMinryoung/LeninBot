# Vector Corpus 재등록 운영 가이드

2026-09-07 `corpus/store.py`의 chunk 정책을 확인했다. 운영 DB의 현재 행수와 저자별 완료 상태는 별도 감사가 필요하다. GPU 호스트의 하드웨어와 가용성은 실행 전에 확인한다.

## Current Runtime Policy

새 ingestion은 `corpus.store.ingest_to_corpus()`를 사용한다. 기본 chunk 정책은 언어별로 분리한다.

| Text type | Default chunk size | Default overlap | Notes |
|---|---:|---:|---|
| English/default | 3000 chars | 300 chars | Marxist classics, English translations |
| Korean | 1800 chars | 200 chars | Korean analysis/commentary; Hangul is denser per char |

`CORPUS_EN_CHUNK_SIZE`, `CORPUS_EN_CHUNK_OVERLAP`, `CORPUS_KO_CHUNK_SIZE`, `CORPUS_KO_CHUNK_OVERLAP`, `CORPUS_EMBED_BATCH_SIZE` may override these defaults.

For GPU embedding, start with:

```
CORPUS_EMBED_BATCH_SIZE=16
```

Raise to 32 only after confirming VRAM and request latency are stable.

## Required Metadata

Every new row should include:

- `layer`
- `author`
- `title`
- `source`
- `source_url` when available
- `public_url` when available
- `year` when known
- `language`
- `chunk_size`
- `chunk_overlap`
- `chunk_index`
- `chunk_count`

Use canonical author names aligned with KG/tool usage:

| Author family | Canonical `metadata.author` |
|---|---|
| Karl Marx / Friedrich Engels | `Marx & Engels` |
| V. I. Lenin | `Lenin` |
| Rosa Luxemburg | `Rosa Luxemburg` |
| Leon Trotsky | `Trotsky` |
| Antonio Gramsci | `Gramsci` |
| Mao Zedong | `Mao` |
| J. V. Stalin | `Stalin` |

If a source is a chapter of a larger work, keep both levels explicit:

- `title`: formal work title
- `section_title` or `chapter_title`: chapter/session/part label
- `source`: stable display/source string, e.g. `Marx & Engels: Capital Vol. I — Chapter 10`

Do not store formal work titles only as free-form `source` while leaving `title` empty.

## 재등록 범위

- `core_theory`는 정식 저작을 대상으로 하고 index/abstract/study-guide 페이지를 제외한다.
- Mao 로컬 crawl에는 중복 꼬리가 관찰되었으므로 선별 manifest 없이 전체 재등록하지 않는다.
- `modern_analysis`의 기존 범위는 한국어 단체 문서 `bolky_`, `diamat_`, `uprising_`이며 `arxiv_`, `bis_`, `mxo_`는 제외한다. 범위 변경은 manifest에 명시한다.
- 2026-05-10 인수인계의 저자별 완료 수치와 “Stalin 진행 중” 표기는 현재 상태가 아니다. 아래 질의나 MCP `corpus_metadata_audit`로 재확인한다.

## Recommended Order

1. Build a curated Mao manifest before reingesting Mao. Do not ingest all `docs/theorists/mao_*.txt` files blindly; the local crawl has large repeated-tail artifacts.
2. Keep `modern_analysis` scoped to Korean organization documents unless the
   layer policy is deliberately changed. Do not re-add arXiv/BIS/MXO material
   without a curated manifest.
3. If additional Marx/Lenin/etc. source files are added later, use the safe pattern below and skip index/abstract/study-guide pages.

## Safe Reingestion Pattern

For each author or source family:

1. Build a manifest first. Include canonical `author`, formal `title`, `year`, `source_url`, local file/page URL, and language.
2. Dry-run extraction. Print estimated character length and expected chunk count.
3. Insert a small sample into `lenin_corpus`.
4. Verify with:

```
SELECT metadata->>'author', metadata->>'title', metadata->>'chunk_size',
       count(*) AS chunks, count(distinct metadata->>'source') AS sources
  FROM lenin_corpus
 WHERE metadata->>'layer' = 'core_theory'
 GROUP BY 1,2,3
 ORDER BY chunks DESC
 LIMIT 50;
```

5. Only then delete old rows for that author/source family.
6. Reingest the full manifest.

GPU 전용 ingestion은 BGE-M3를 CUDA에 로드한 뒤 `corpus.embeddings.set_shared_embeddings()`로 공유할 수 있다. 과거 `temp_dev/` 일회성 helper를 배포된 진입점으로 간주하지 않는다. 실행 스크립트와 manifest를 먼저 확인한다.

Prefer deleting narrowly by `layer`, canonical `author`, and either `source_url` or manifest source IDs. Avoid broad deletes unless the manifest is complete and tested.

## Validation Queries

Check missing titles:

```
SELECT metadata->>'layer', metadata->>'author',
       count(*) FILTER (WHERE coalesce(metadata->>'title','') = '') AS missing_title,
       count(*) AS chunks
  FROM lenin_corpus
 GROUP BY 1,2
 ORDER BY missing_title DESC;
```

Check chunk-size distribution:

```
SELECT metadata->>'layer', metadata->>'author',
       coalesce(metadata->>'chunk_size','missing') AS chunk_size,
       count(*) AS chunks,
       count(distinct metadata->>'source') AS sources
  FROM lenin_corpus
 GROUP BY 1,2,3
 ORDER BY chunks DESC;
```

Check whether search headers will be useful:

```
SELECT metadata->>'author' AS author,
       metadata->>'title' AS title,
       metadata->>'source' AS source,
       metadata->>'source_url' AS source_url
  FROM lenin_corpus
 WHERE metadata->>'layer' = 'core_theory'
 ORDER BY random()
 LIMIT 20;
```

## Notes

- Existing rows without `chunk_size` may be usable, but they are hard to audit and hard to compare.
- For very long works, decide source granularity before embedding. Whole work gives better document-level context; chapter-level source gives easier citation and reingestion. In either case, preserve both formal title and section/chapter title in metadata.
- `vector_search` now supports optional `author`, `title`, `year`, and `keywords` filters. Good metadata directly improves retrieval quality.
