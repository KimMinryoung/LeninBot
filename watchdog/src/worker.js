// Off-VM watchdog for leninbot, running on Cloudflare Workers.
//
// Everything that alerts today runs on the main VM and reports over Telegram,
// so the one failure it cannot report is the VM itself going away. This Worker
// lives outside that blast radius. It does two things:
//
//   1. cron — fetch the public site and alert when the origin stops answering.
//      On 2026-08-01 a firewall change dropped the origin's 80/443 and the site
//      served 522 for several minutes with no alert at all; Cloudflare's cache
//      made it look healthy in a browser the whole time. Hence the cache-buster
//      and `cache: 'no-store'` below — checking the cached copy would have
//      reproduced exactly that blind spot.
//
//   2. /ping/<token>/<job> — a dead-man's switch for the VM's systemd timers.
//      A job that succeeds pings; the cron alerts when a ping is overdue. This
//      is what catches "the VM is gone", because a dead VM stops pinging.
//
// State lives in KV and alerts are edge-triggered: one message when a check
// goes bad, one when it recovers, silence in between. That also keeps writes
// far below the KV free-tier allowance — the cron reads state every run but
// only writes when something actually changed.

const SITE_URL = "https://cyber-lenin.com/";
const SITE_TIMEOUT_MS = 20000;

// A 200 is not proof the site works. On 2026-07-28 the frontend kept answering
// 200 while every post had vanished — its container still pointed at the old
// database. Status-only monitoring would have called that healthy, so assert on
// the body too. Thresholds are deliberately loose (the homepage currently shows
// 4 reports and 1 term in ~33KB); a redesign should not trip these, and when
// one does fire the message names the exact assertion so it is obvious whether
// the site broke or the check needs updating.
const DB_LINK_RE = /href="\/(?:reports|commulingo)\//g;
const CONTENT_CHECKS = [
  {
    name: "본문 길이",
    ok: (body) => body.length >= 5000,
    detail: (body) => `${body.length} bytes (기준 5000)`,
  },
  {
    name: "타이틀",
    ok: (body) => /<title>[^<]*Cyber-Lenin/i.test(body),
    detail: () => "<title>에 Cyber-Lenin 없음",
  },
  {
    name: "DB 유래 링크",
    ok: (body) => (body.match(DB_LINK_RE) || []).length >= 3,
    detail: (body) => `${(body.match(DB_LINK_RE) || []).length}개 (기준 3)`,
  },
];

// periodMin is how often the job is supposed to report; graceMin is how long
// past that we stay quiet before calling it overdue. replication-health runs
// every 15 minutes, which makes it the de facto VM heartbeat.
const JOBS = {
  "replication-health": { periodMin: 15, graceMin: 45, label: "복제 상태 점검" },
  "main-backup": { periodMin: 1440, graceMin: 180, label: "메인 DB 백업" },
  "writer-backup": { periodMin: 1440, graceMin: 180, label: "writer DB 백업" },
  "kg-backup": { periodMin: 1440, graceMin: 180, label: "KG 백업" },
};

async function sendTelegram(env, text) {
  if (!env.TELEGRAM_BOT_TOKEN || !env.TELEGRAM_CHAT_ID) return false;
  try {
    const resp = await fetch(
      `https://api.telegram.org/bot${env.TELEGRAM_BOT_TOKEN}/sendMessage`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ chat_id: env.TELEGRAM_CHAT_ID, text }),
      },
    );
    return resp.ok;
  } catch (_) {
    return false;
  }
}

/**
 * Alert only when `healthy` differs from what we last recorded for `key`.
 * Returns true when a message was actually sent.
 */
async function transition(env, key, healthy, badText, okText) {
  const stateKey = `state:${key}`;
  const previous = (await env.WATCHDOG.get(stateKey)) ?? "ok";
  const current = healthy ? "ok" : "bad";
  if (previous === current) return false;
  await env.WATCHDOG.put(stateKey, current);
  await sendTelegram(env, healthy ? okText : badText);
  return true;
}

async function checkSite(env) {
  // Cache-buster plus no-store: a cached 200 would hide a dead origin, which is
  // the exact way the 2026-08-01 outage stayed invisible.
  const url = `${SITE_URL}?watchdog=${Date.now()}`;
  let status = 0;
  let detail = "";
  let healthy = false;
  try {
    const resp = await fetch(url, {
      cache: "no-store",
      redirect: "manual",
      signal: AbortSignal.timeout(SITE_TIMEOUT_MS),
      headers: { "User-Agent": "leninbot-watchdog/1" },
    });
    status = resp.status;
    if (status < 200 || status >= 300) {
      // 3xx counts as unhealthy here: the site serves 200 directly, so a
      // redirect means something changed that a human should look at.
      detail = `HTTP ${status}`;
    } else {
      const body = await resp.text();
      const failed = CONTENT_CHECKS.filter((c) => !c.ok(body));
      healthy = failed.length === 0;
      detail = healthy
        ? `HTTP ${status}, 콘텐츠 검사 통과`
        : `HTTP ${status}이지만 콘텐츠 이상 — ` +
          failed.map((c) => `${c.name}: ${c.detail(body)}`).join(", ");
    }
  } catch (err) {
    detail = `요청 실패: ${err && err.name ? err.name : "error"}`;
  }
  await transition(
    env,
    "site",
    healthy,
    `🔴 cyber-lenin.com 응답 이상\n\n${detail}\n\n` +
      `522면 Cloudflare가 origin에 못 닿는 것입니다 — 방화벽 인바운드 80/443과 nginx를 보세요.\n` +
      `콘텐츠 이상이면 응답은 오는데 DB에서 글이 안 나오는 것입니다 — ` +
      `frontend 컨테이너의 DB 접속과 leninbot-pg를 보세요.`,
    `🟢 cyber-lenin.com 복구 (${detail})`,
  );
  return { healthy, detail };
}

async function checkJobs(env) {
  const now = Date.now();
  const results = [];
  for (const [job, spec] of Object.entries(JOBS)) {
    const last = await env.WATCHDOG.get(`ping:${job}`);
    const deadlineMs = (spec.periodMin + spec.graceMin) * 60_000;
    // A job that has never pinged is not yet overdue — otherwise deploying the
    // Worker before wiring the units would alert on all four immediately.
    const healthy = last === null || now - Number(last) <= deadlineMs;
    const ageMin = last === null ? null : Math.round((now - Number(last)) / 60_000);
    results.push({ job, healthy, ageMin });
    await transition(
      env,
      `job:${job}`,
      healthy,
      `🔴 ${spec.label}(${job}) 무소식 ${ageMin}분\n\n` +
        `기대 주기 ${spec.periodMin}분 + 유예 ${spec.graceMin}분을 넘겼습니다.\n` +
        `잡이 실패했거나, 메인 VM이 응답하지 않습니다.`,
      `🟢 ${spec.label}(${job}) 정상 복귀`,
    );
  }
  return results;
}

export default {
  async fetch(request, env) {
    const { pathname } = new URL(request.url);

    // /ping/<token>/<job> — the token keeps strangers from suppressing alerts
    // by pinging on the VM's behalf.
    const ping = pathname.match(/^\/ping\/([^/]+)\/([^/]+)\/?$/);
    if (ping) {
      const [, token, job] = ping;
      if (token !== env.PING_TOKEN) return new Response("forbidden", { status: 403 });
      if (!(job in JOBS)) return new Response("unknown job", { status: 404 });
      await env.WATCHDOG.put(`ping:${job}`, String(Date.now()));
      return new Response("ok\n");
    }

    // /status/<token> — manual inspection without digging through KV.
    const status = pathname.match(/^\/status\/([^/]+)\/?$/);
    if (status) {
      if (status[1] !== env.PING_TOKEN) return new Response("forbidden", { status: 403 });
      const now = Date.now();
      const jobs = {};
      for (const [job, spec] of Object.entries(JOBS)) {
        const last = await env.WATCHDOG.get(`ping:${job}`);
        jobs[job] = {
          lastPingIso: last === null ? null : new Date(Number(last)).toISOString(),
          ageMinutes: last === null ? null : Math.round((now - Number(last)) / 60_000),
          overdueAfterMinutes: spec.periodMin + spec.graceMin,
          state: (await env.WATCHDOG.get(`state:job:${job}`)) ?? "ok",
        };
      }
      return Response.json({
        site: { state: (await env.WATCHDOG.get("state:site")) ?? "ok", url: SITE_URL },
        jobs,
      });
    }

    return new Response("not found", { status: 404 });
  },

  // Awaited directly rather than handed to ctx.waitUntil(): a throw then marks
  // the cron invocation as failed and shows up in the Workers dashboard, which
  // matters for a watchdog — one that dies quietly is worse than none.
  async scheduled(_event, env, _ctx) {
    await checkSite(env);
    await checkJobs(env);
  },
};
