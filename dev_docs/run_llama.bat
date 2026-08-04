@echo off
title Qwopus3.5-9B-v3 LLM Server (Q6_K)

REM Qwopus is distilled to reason inside <think> tags — do NOT pass enable_thinking=false.
REM Context 98304: Q6_K (7.36GB) + q4_0 KV cache ~3.5GB fits in 12GB VRAM with headroom.

C:\Users\DESKTOP\llama-cpp\bin\llama-server.exe ^
  -m C:\Users\DESKTOP\llama-cpp\models\Qwopus3.5-9B-v3.Q6_K.gguf ^
  --host 0.0.0.0 --port 8080 ^
  -ngl 99 -c 98304 ^
  --flash-attn on ^
  --cache-type-k q4_0 ^
  --cache-type-v q4_0 ^
  --ubatch-size 512 ^
  --threads 4 ^
  --parallel 1 ^
  --cont-batching ^
  --no-mmap ^
  --slots ^
  --jinja
pause
