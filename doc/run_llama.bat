@echo off
title Qwen3.5-9B LLM Server (Q4_K_M)

C:\Users\DESKTOP\llama-cpp\bin\llama-server.exe ^
  -m C:\Users\DESKTOP\llama-cpp\models\Qwen3.5-9B-Q4_K_M.gguf ^
  --host 0.0.0.0 --port 8080 ^
  -ngl 99 -c 131072 ^
  --flash-attn on ^
  --cache-type-k q4_0 ^
  --cache-type-v q4_0 ^
  --ubatch-size 512 ^
  --threads 4 ^
  --parallel 1 ^
  --cont-batching ^
  --no-mmap ^
  --slots ^
  --jinja ^
  --chat-template-kwargs "{\"enable_thinking\":false}"
pause