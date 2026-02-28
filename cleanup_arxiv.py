"""cleanup_arxiv.py
Deletes all arxiv_* data from:
  1. Supabase DB (lennin_corpus table)
  2. processed_files.txt log
  3. Local docs/modern_analysis/arxiv_*.txt files

Run BEFORE re-crawling with new targeted arXiv queries.
"""
import os
import glob
from dotenv import load_dotenv
from supabase.client import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Same table name as update_knowledge.py
TABLE = "len" + "in_corpus"   # -> "lennin_corpus"  (Lenin + _corpus)
MODERN_DIR = "./docs/modern_analysis"
LOG_FILE = "processed_files.txt"


def extract_title(fpath):
    """Extract the Title: header from a file (used as 'source' in DB metadata)."""
    try:
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith("Title:") and line[6:].strip():
                    return line[6:].strip()
                if line and not line.startswith("Source:") and not line.startswith("Title:"):
                    break  # past header
    except Exception:
        pass
    return os.path.splitext(os.path.basename(fpath))[0]  # fallback: filename stem


def main():
    arxiv_files = glob.glob(os.path.join(MODERN_DIR, "arxiv_*.txt"))
    print(f"arxiv 파일 발견: {len(arxiv_files)}개")
    print(f"테이블: {TABLE}")

    # ── Step 1: DB 삭제 ──────────────────────────────────────────────────────
    print("\n[1/3] Supabase DB에서 arxiv 행 삭제 중...")
    deleted = 0
    failed = []
    for i, fpath in enumerate(arxiv_files, 1):
        if i % 100 == 0:
            print(f"  진행: {i}/{len(arxiv_files)} ({deleted}건 삭제)")
        title = extract_title(fpath)
        try:
            # Use @> (contains) operator for JSONB filtering — avoids PGRST205
            supabase.table(TABLE).delete().contains(
                "metadata", {"source": title, "layer": "modern_analysis"}
            ).execute()
            deleted += 1
        except Exception as e:
            failed.append((title[:60], str(e)[:80]))

    print(f"  완료: {deleted}건 삭제 요청, {len(failed)}건 실패")
    if failed:
        print("  실패 샘플 (첫 5건):")
        for t, e in failed[:5]:
            print(f"    - {t}: {e}")

    # ── Step 2: processed_files.txt 정리 ─────────────────────────────────────
    print("\n[2/3] processed_files.txt에서 arxiv 항목 제거 중...")
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()

        kept = [
            l for l in lines
            if not (
                'modern_analysis' in l
                and os.path.basename(l.replace('\\', '/')).startswith('arxiv_')
            )
        ]
        removed = len(lines) - len(kept)

        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            f.write('\n'.join(kept))
            if kept:
                f.write('\n')

        print(f"  {removed}개 항목 제거, {len(kept)}개 유지")
    else:
        print("  processed_files.txt 없음 — 건너뜀")

    # ── Step 3: 로컬 파일 삭제 ───────────────────────────────────────────────
    print("\n[3/3] 로컬 arxiv 파일 삭제 중...")
    file_count = 0
    for fpath in arxiv_files:
        try:
            os.remove(fpath)
            file_count += 1
        except Exception as e:
            print(f"  삭제 실패: {fpath} - {e}")
    print(f"  {file_count}개 파일 삭제 완료")

    print("\n✅ 정리 완료!")
    print("다음 명령으로 재크롤링 및 재적재하세요:")
    print("  python crawler_modern.py --source arxiv")
    print("  python update_knowledge.py --layer modern_analysis --source-dir ./docs/modern_analysis")


if __name__ == "__main__":
    main()
