#!/bin/bash
# DeepSeek 사료 번역을 credstore 자격 증명과 함께 돌린다.
#
# 번역 CLI는 systemd-run으로 자격 증명을 받아야 하는데, 그 명령줄이 길어
# 터미널에서 줄바꿈되면 조각조각 실행되어 버린다. 그래서 한 줄로 부를 수
# 있게 감싸 둔다.
#
#   sudo /home/grass/leninbot/scripts/run-archival-translation.sh --spec <id>
#
# --plan 은 키가 필요 없으므로 sudo 없이 venv/bin/python으로 직접 부르면 된다.
set -euo pipefail

exec systemd-run --pipe --quiet --collect \
  -p User=grass \
  -p WorkingDirectory=/home/grass/leninbot \
  -p LoadCredentialEncrypted=deepseek_api_key:/etc/credstore.encrypted/deepseek_api_key.cred \
  /home/grass/leninbot/venv/bin/python \
  /home/grass/leninbot/scripts/translate_archival_documents.py "$@"
