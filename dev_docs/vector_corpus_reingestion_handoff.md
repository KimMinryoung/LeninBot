# Vector Corpus Reingestion Handoff

최종 확인 기준: 2026-05-10 운영 DB 관찰.

이 문서는 Stalin 외 `lenin_corpus`를 Windows GPU PC에서 재등록하기 위한 인수인계다. 해당 PC는 BGE-M3가 이미 설치되어 있고 12GB VRAM이 있어 대량 embedding 작업에 더 적합하다.

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

## Current Cleanup Targets

### Already handled on server

- Mao was deleted from `core_theory` because its old corpus was extremely over-chunked and lacked reliable metadata.
- Stalin is being reingested on the server with English/default `3000/300` chunks and corrected metadata.

### Needs Windows GPU reingestion or metadata repair

`core_theory` non-Stalin rows generally have `metadata.source` but lack:

- `metadata.title`
- `metadata.chunk_size`
- `metadata.chunk_overlap`
- source URL/public URL in many cases

Observed non-Stalin core-theory counts before cleanup:

| Author | Chunks | Sources | Issue |
|---|---:|---:|---|
| Marx & Engels | 25,151 | 135 | missing `title`, missing chunk-size metadata; some sources are chapter-level with very high row counts |
| Lenin | 10,524 | 335 | missing `title`, missing chunk-size metadata |
| Rosa Luxemburg | 5,880 | 180 | missing `title`, missing chunk-size metadata |
| Trotsky | 4,397 | 76 | missing `title`, missing chunk-size metadata |
| Gramsci | 831 | 48 | missing `title`, missing chunk-size metadata |

`modern_analysis` is also mixed-generation:

- most rows lack `title` and chunk-size metadata
- a small newer subset has `chunk_size=900`, too small for long Korean analysis
- reingest Korean long-form analysis with Korean/default `1800/200`

## Recommended Order

1. Reingest Mao into `core_theory` from clean source files/pages.
2. Reingest or metadata-repair Marx & Engels, prioritizing works that appear as huge chapter-level sources:
   - `Capital Vol. I`
   - `Grundrisse`
   - `The German Ideology`
   - `The Civil War in France`
   - `Anti-Dühring`
3. Reingest Lenin with formal work titles and chapter/session labels separated.
4. Reingest Rosa Luxemburg, Trotsky, Gramsci.
5. Audit `modern_analysis`; reingest Korean long-form material with `language="ko"` and `1800/200` chunks.

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
