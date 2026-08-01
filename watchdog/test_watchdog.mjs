// Exercise the watchdog Worker against a fake KV and a fake network.
import worker from "/home/grass/leninbot/watchdog/src/worker.js";

const PING_TOKEN = "testtoken";
let kv, sent, siteStatus, siteThrows;

function makeEnv() {
  const store = new Map();
  kv = store;
  return {
    WATCHDOG: {
      get: async (k) => (store.has(k) ? store.get(k) : null),
      put: async (k, v) => void store.set(k, v),
    },
    TELEGRAM_BOT_TOKEN: "bot:token",
    TELEGRAM_CHAT_ID: "12345",
    PING_TOKEN,
  };
}

globalThis.fetch = async (url, init) => {
  const u = String(url);
  if (u.startsWith("https://api.telegram.org/")) {
    sent.push(JSON.parse(init.body).text);
    return new Response("{}", { status: 200 });
  }
  if (u.startsWith("https://cyber-lenin.com/")) {
    if (!u.includes("watchdog=")) throw new Error("site fetched without cache-buster!");
    if (siteThrows) throw Object.assign(new Error("timeout"), { name: "TimeoutError" });
    return new Response("body", { status: siteStatus });
  }
  throw new Error("unexpected fetch: " + u);
};

const ctx = { waitUntil: (p) => p };
let pass = 0, fail = 0;
function check(name, cond, detail = "") {
  if (cond) { pass++; console.log(`  [PASS] ${name}`); }
  else { fail++; console.log(`  [FAIL] ${name}  ${detail}`); }
}
const run = async (env) => { sent = []; await worker.scheduled({}, env, ctx); return sent; };
const req = (path) => new Request("https://w.example.com" + path);

console.log("═══ ping 엔드포인트 ═══");
{
  const env = makeEnv();
  const bad = await worker.fetch(req(`/ping/wrong/main-backup`), env);
  check("잘못된 토큰 → 403", bad.status === 403, `got ${bad.status}`);
  const unk = await worker.fetch(req(`/ping/${PING_TOKEN}/nope`), env);
  check("모르는 job → 404", unk.status === 404, `got ${unk.status}`);
  const ok = await worker.fetch(req(`/ping/${PING_TOKEN}/main-backup`), env);
  check("정상 핑 → 200", ok.status === 200, `got ${ok.status}`);
  check("KV에 기록됨", kv.has("ping:main-backup"));
  const forbid = await worker.fetch(req(`/status/wrong`), env);
  check("status 잘못된 토큰 → 403", forbid.status === 403, `got ${forbid.status}`);
}

console.log("═══ 사이트 감시 ═══");
{
  const env = makeEnv();
  siteThrows = false; siteStatus = 200;
  check("정상이면 조용함", (await run(env)).length === 0);

  siteStatus = 522;
  let msgs = await run(env);
  check("522 → 알림 1건", msgs.length === 1 && msgs[0].includes("522"), JSON.stringify(msgs));

  msgs = await run(env);
  check("계속 죽어 있어도 재알림 없음", msgs.length === 0, JSON.stringify(msgs));

  siteStatus = 200;
  msgs = await run(env);
  check("복구 시 복구 알림", msgs.length === 1 && msgs[0].includes("🟢"), JSON.stringify(msgs));

  siteThrows = true;
  msgs = await run(env);
  check("타임아웃도 이상으로 감지", msgs.length === 1 && msgs[0].includes("🔴"), JSON.stringify(msgs));
  siteThrows = false; siteStatus = 200;
}

console.log("═══ 데드맨 스위치 ═══");
{
  const env = makeEnv();
  siteStatus = 200; siteThrows = false;
  check("핑 이력이 없으면 알림 없음 (배포 직후 오탐 방지)", (await run(env)).length === 0);

  await worker.fetch(req(`/ping/${PING_TOKEN}/replication-health`), env);
  check("방금 핑했으면 정상", (await run(env)).length === 0);

  // 15분 주기 + 45분 유예 = 60분. 61분 전으로 되돌린다.
  kv.set("ping:replication-health", String(Date.now() - 61 * 60_000));
  let msgs = await run(env);
  check("61분 무소식 → 알림", msgs.length === 1 && msgs[0].includes("replication-health"), JSON.stringify(msgs));
  check("알림에 경과 분 포함", msgs[0] && /\d+분/.test(msgs[0]));

  check("반복 알림 없음", (await run(env)).length === 0);

  await worker.fetch(req(`/ping/${PING_TOKEN}/replication-health`), env);
  msgs = await run(env);
  check("핑 재개 → 복구 알림", msgs.length === 1 && msgs[0].includes("🟢"), JSON.stringify(msgs));

  // 일일 잡: 24시간 + 3시간 유예 = 27시간. 26시간은 아직 유예 안.
  kv.set("ping:main-backup", String(Date.now() - 26 * 60 * 60_000));
  check("일일 잡 26시간 → 아직 유예 내", (await run(env)).length === 0);
  kv.set("ping:main-backup", String(Date.now() - 28 * 60 * 60_000));
  msgs = await run(env);
  check("일일 잡 28시간 → 알림", msgs.length === 1 && msgs[0].includes("main-backup"), JSON.stringify(msgs));
}

console.log("═══ status 엔드포인트 ═══");
{
  const env = makeEnv();
  await worker.fetch(req(`/ping/${PING_TOKEN}/kg-backup`), env);
  const resp = await worker.fetch(req(`/status/${PING_TOKEN}`), env);
  const body = await resp.json();
  check("200 + JSON", resp.status === 200 && body.jobs && body.site, JSON.stringify(body).slice(0, 120));
  check("kg-backup 최근 핑 기록됨", body.jobs["kg-backup"].ageMinutes === 0, JSON.stringify(body.jobs["kg-backup"]));
  check("미핑 잡은 null", body.jobs["main-backup"].lastPingIso === null);
}

console.log(`\n═══════ ${pass} passed, ${fail} failed ═══════`);
process.exit(fail ? 1 : 0);
