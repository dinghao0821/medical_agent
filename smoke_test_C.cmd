@echo off
REM ============================================================
REM  C-tier smoke test: Docker multi-worker + Redis + nginx SSE
REM  Verifies P1 acceptance items that cannot run on Windows host:
REM    - multi-worker startup (gunicorn + UvicornWorker)
REM    - cross-worker session sharing via Redis checkpointer
REM    - session isolation (no cross-talk)
REM    - non-blocking under concurrency
REM    - SSE streaming through nginx (proxy_buffering off)
REM
REM  Prereqs: `docker compose up --build -d` already running, .env filled.
REM  Usage:   smoke_test_C.cmd
REM ============================================================
setlocal enabledelayedexpansion
chcp 65001 >nul

set "BASE=http://localhost:8080"
set /a PASS=0
set /a FAIL=0
set /a WARN=0

echo ============================================================
echo   C-tier Smoke Test  (target: %BASE%)
echo ============================================================
echo.

REM ---- prerequisite: curl available ----
where curl >nul 2>&1
if errorlevel 1 (
  echo [ERROR] curl not found. Windows 10/11 ships curl; please install it.
  exit /b 1
)

REM ---- prerequisite: docker compose reachable ----
docker compose version >nul 2>&1
if errorlevel 1 (
  echo [ERROR] "docker compose" not available. Is Docker Desktop running?
  exit /b 1
)

REM ============================================================
echo [1/6] Multi-worker startup check (gunicorn UvicornWorker)
REM ============================================================
docker compose logs app 2>nul | findstr /i "Booting worker" >nul
if errorlevel 1 (
  echo   [WARN] Could not find "Booting worker" in app logs.
  echo          Check manually: docker compose logs app ^| findstr /i worker
  set /a WARN+=1
) else (
  for /f %%c in ('docker compose logs app 2^>nul ^| findstr /i /c:"Booting worker" ^| find /c /v ""') do set WK=%%c
  echo   [PASS] Detected !WK! worker boot line^(s^).
  set /a PASS+=1
)
echo.

REM ============================================================
echo [2/6] Health check via nginx
REM ============================================================
for /f %%i in ('curl -s -o health.txt -w "%%{http_code}" %BASE%/health') do set CODE=%%i
if "!CODE!"=="200" (
  echo   [PASS] /health returned 200.
  set /a PASS+=1
) else (
  echo   [FAIL] /health returned !CODE! ^(expected 200^). Is the stack up?
  set /a FAIL+=1
)
echo.

REM ============================================================
echo [3/6] Cross-worker session sharing (Redis checkpointer)
REM ============================================================
if exist share.txt del share.txt
echo {"query":"My name is Alex. Please remember my name."}> q_set.json
echo {"query":"What is my name? Reply with just the name."}> q_get.json

curl -s -c cookies.txt -b cookies.txt -X POST %BASE%/chat -H "Content-Type: application/json" --data-binary "@q_set.json" >nul
curl -s -c cookies.txt -b cookies.txt -X POST %BASE%/chat -H "Content-Type: application/json" --data-binary "@q_get.json" -o share.txt

findstr /i "Alex" share.txt >nul
if errorlevel 1 (
  echo   [FAIL] Follow-up did NOT recall "Alex" -> session state not shared across workers.
  echo          Response snippet:
  type share.txt
  set /a FAIL+=1
) else (
  echo   [PASS] Follow-up recalled "Alex" -> Redis-backed session sharing works.
  set /a PASS+=1
)
echo.

REM ============================================================
echo [4/6] Session isolation (no cross-talk between sessions)
REM ============================================================
if exist iso.txt del iso.txt
if exist other.txt del other.txt
curl -s -c other.txt -b other.txt -X POST %BASE%/chat -H "Content-Type: application/json" --data-binary "@q_get.json" -o iso.txt

findstr /i "Alex" iso.txt >nul
if errorlevel 1 (
  echo   [PASS] Fresh session does NOT know "Alex" -> sessions are isolated.
  set /a PASS+=1
) else (
  echo   [FAIL] Fresh session leaked "Alex" -> sessions are NOT isolated!
  set /a FAIL+=1
)
echo.

REM ============================================================
echo [5/6] Redis contains checkpoint state
REM ============================================================
for /f %%i in ('docker compose exec -T redis redis-cli DBSIZE 2^>nul') do set DBSIZE=%%i
if "!DBSIZE!"=="" (
  echo   [WARN] Could not read Redis DBSIZE. Check: docker compose exec redis redis-cli DBSIZE
  set /a WARN+=1
) else (
  echo   Redis DBSIZE = !DBSIZE!
  if "!DBSIZE!"=="0" (
    echo   [WARN] Redis is empty -- checkpointer may not be using Redis. Verify CHECKPOINTER_BACKEND=redis.
    set /a WARN+=1
  ) else (
    echo   [PASS] Redis holds checkpoint keys.
    set /a PASS+=1
  )
)
echo.

REM ============================================================
echo [6/6] SSE streaming through nginx
REM ============================================================
if exist stream.txt del stream.txt
echo {"query":"In one sentence, what is a migraine?"}> q_stream.json
curl -sN -X POST %BASE%/chat/stream -H "Content-Type: application/json" --data-binary "@q_stream.json" -o stream.txt

findstr /c:"\"type\": \"done\"" /c:"\"type\":\"done\"" /c:"done" stream.txt >nul
if errorlevel 1 (
  echo   [FAIL] No SSE terminal "done" event captured.
  echo          Raw output:
  type stream.txt
  set /a FAIL+=1
) else (
  findstr /c:"token" stream.txt >nul
  if errorlevel 1 (
    echo   [WARN] Got "done" but no "token" events -- streaming may be degraded.
    set /a WARN+=1
  ) else (
    echo   [PASS] Received incremental "token" events and a "done" event.
    set /a PASS+=1
  )
)
echo.

REM ---- cleanup temp files ----
del /q q_set.json q_get.json q_stream.json health.txt share.txt iso.txt stream.txt cookies.txt other.txt 2>nul

echo ============================================================
echo   RESULT:  PASS=%PASS%  FAIL=%FAIL%  WARN=%WARN%
echo ============================================================
if %FAIL% GTR 0 (
  echo   C-tier NOT fully passed. Review [FAIL] items above.
  exit /b 1
) else (
  echo   C-tier smoke checks passed ^(review any [WARN] items^).
  echo   Note: concurrency / "slow request does not block" is best verified
  echo   manually ^(fire a slow /upload while sending quick /chat requests^).
  exit /b 0
)
