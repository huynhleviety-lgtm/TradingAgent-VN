"use client";

import { useCallback, useEffect, useState } from "react";
import { VT323 } from "next/font/google";
import { getQlibScores, getSystemStatus, startQlibRefresh, syncMarket } from "@/lib/api";
import type { QlibScores, SystemStatus } from "@/lib/types";
import { SectionHeader } from "@/components/SectionHeader";
import { StatBlock } from "@/components/StatBlock";

const vt323 = VT323({ weight: "400", subsets: ["latin"], variable: "--font-vt323" });

function StatusBadge({ ok, warn }: { ok: boolean; warn?: boolean }) {
  if (ok) return <span className="badge badge-buy">OK</span>;
  if (warn) return <span className="badge badge-warm">Cần xử lý</span>;
  return <span className="badge badge-sell">Lỗi</span>;
}

function Card({
  title,
  tag,
  badge,
  message,
  children,
  actions,
}: {
  title: string;
  tag: string;
  badge: React.ReactNode;
  message: string;
  children: React.ReactNode;
  actions?: React.ReactNode;
}) {
  return (
    <section className="pixel-border bg-white p-5 flex flex-col gap-4 min-w-0">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2 min-w-0">
          <span className="tagline">{tag}</span>
          <h3 className={`${vt323.className} text-2xl uppercase tracking-[0.06em] text-black leading-none`}>
            {title}
          </h3>
        </div>
        {badge}
      </div>
      <div className="grid grid-cols-2 gap-2">{children}</div>
      <p className="text-[12px] font-mono text-[#555555] leading-snug break-words">{message}</p>
      {actions && <div className="flex flex-wrap gap-2 mt-auto">{actions}</div>}
    </section>
  );
}

function fmtIc(v: number | null | undefined): string {
  return typeof v === "number" ? v.toFixed(3) : "—";
}

export default function SystemPage() {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [scores, setScores] = useState<QlibScores | null>(null);
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState<"" | "status" | "sync" | "qlib">("");
  const [note, setNote] = useState("");

  const load = useCallback(
    () =>
      Promise.all([getSystemStatus(), getQlibScores(100)])
        .then(([s, q]) => {
          setStatus(s);
          setScores(q);
          setErr("");
        })
        .catch((e) => setErr(e instanceof Error ? e.message : String(e))),
    [],
  );

  useEffect(() => {
    Promise.all([getSystemStatus(), getQlibScores(100)])
      .then(([s, q]) => {
        setStatus(s);
        setScores(q);
      })
      .catch((e) => setErr(e instanceof Error ? e.message : String(e)));
  }, []);

  // While Qlib runs in the background, poll until it finishes.
  const qlibRunning = status?.qlib.refresh.running ?? false;
  useEffect(() => {
    if (!qlibRunning) return;
    const id = setInterval(load, 3000);
    return () => clearInterval(id);
  }, [qlibRunning, load]);

  async function refreshStatus() {
    setBusy("status");
    await load();
    setBusy("");
  }

  async function syncStale() {
    if (!status) return;
    setBusy("sync");
    setNote("");
    try {
      await syncMarket(status.market_data.stale_tickers);
      setNote(`Đã đồng bộ ${status.market_data.stale_tickers.length} mã.`);
      await load();
    } catch (e) {
      setNote(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy("");
    }
  }

  async function runQlib() {
    setBusy("qlib");
    setNote("");
    try {
      const r = await startQlibRefresh();
      setNote(r.started ? "Qlib đang chạy nền (khoảng 1 phút với VN30)…" : r.reason ?? "");
      await load();
    } catch (e) {
      setNote(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy("");
    }
  }

  const llm = status?.llm;
  const md = status?.market_data;
  const ql = status?.qlib;
  const run = scores?.run ?? ql?.last_run ?? null;
  const lastExit = ql?.refresh.exit_code;

  return (
    <div className="w-full px-4 py-8 md:px-8 md:py-10">
      <div className="mx-auto w-full max-w-[1500px] flex flex-col gap-8">
        <SectionHeader
          step="Hệ thống"
          title="Trạng thái"
          trailing={
            <button onClick={refreshStatus} disabled={busy !== ""} className="btn-ghost whitespace-nowrap">
              {busy === "status" ? "Đang kiểm tra…" : "↻ Kiểm tra lại"}
            </button>
          }
        />

        {err && (
          <div className="pixel-border bg-white p-4 text-[13px] font-mono text-[#B91C1C]">
            {err}. Backend (FastAPI, cổng 8000) đã chạy chưa?
          </div>
        )}
        {note && <div className="text-[12px] font-mono text-black">{note}</div>}

        {status && llm && md && ql && (
          <div className="grid gap-6 lg:grid-cols-3">
            <Card
              tag="01"
              title="AI local"
              badge={<StatusBadge ok={llm.ok} warn={llm.reachable} />}
              message={llm.message}
            >
              <StatBlock label="Máy chủ" value={llm.reachable ? (llm.server ?? "on") : "off"} tone={llm.reachable ? "green" : "red"} sub={<span className="break-all">{llm.base_url}</span>} />
              <StatBlock label="Song song" value={llm.concurrency} sub="LLM_CONCURRENCY" />
              <StatBlock label="Analysts" value={llm.configured.analysts} tone={llm.missing.includes(llm.configured.analysts) ? "red" : "default"} />
              <StatBlock label="Debate / CIO" value={llm.configured.cio} tone={llm.missing.includes(llm.configured.cio) ? "red" : "default"} />
            </Card>

            <Card
              tag="02"
              title="Dữ liệu"
              badge={<StatusBadge ok={md.ok} warn={md.stocks > 0} />}
              message={md.message}
              actions={
                md.stale_tickers.length > 0 ? (
                  <button onClick={syncStale} disabled={busy !== ""} className="btn-primary">
                    {busy === "sync" ? "Đang đồng bộ…" : `Đồng bộ ${md.stale_tickers.length} mã`}
                  </button>
                ) : null
              }
            >
              <StatBlock label="Cổ phiếu" value={md.stocks} sub={md.indices.join(", ") || "chưa có chỉ số"} />
              <StatBlock
                label="Phiên mới nhất"
                value={md.latest_date ? md.latest_date.slice(5) : "—"}
                tone={md.ok ? "green" : "red"}
                sub={md.stale_days != null ? `${md.stale_days} ngày trước` : ""}
              />
              <StatBlock label="Chỉ số tài chính" value={md.ratio_tickers} sub="mã có ratios" />
              <StatBlock label="Cache" value={md.cache_exists ? "có" : "chưa"} sub="vn-market-data" />
            </Card>

            <Card
              tag="03"
              title="Qlib"
              badge={<StatusBadge ok={ql.ok} warn={ql.python_ok && ql.repo_ok} />}
              message={
                ql.refresh.running
                  ? "Đang chạy Alpha158 + LightGBM…"
                  : lastExit != null && lastExit !== 0
                    ? `Lần chạy gần nhất lỗi (mã ${lastExit}) — xem log của backend.`
                    : ql.message
              }
              actions={
                <button
                  onClick={runQlib}
                  disabled={busy !== "" || ql.refresh.running || !ql.python_ok || !ql.repo_ok}
                  className="btn-primary"
                >
                  {ql.refresh.running ? "Đang chạy…" : "Chạy Qlib"}
                </button>
              }
            >
              <StatBlock
                label="Rank IC ngoài mẫu"
                value={fmtIc(run?.rank_ic_mean)}
                tone={(run?.rank_ic_mean ?? 0) >= 0.02 ? "green" : "red"}
                sub={run?.test_days ? `${run.test_days} phiên kiểm định` : ""}
              />
              <StatBlock label="IC IR" value={fmtIc(run?.rank_ic_ir)} />
              <StatBlock label="Huấn luyện đến" value={run?.train_end ?? "—"} sub={run?.valid_end ? `valid → ${run.valid_end}` : ""} />
              <StatBlock label="Lần chạy" value={run ? new Date(run.run_at).toLocaleDateString("vi-VN") : "—"} />
            </Card>
          </div>
        )}

        <div>
          <SectionHeader
            step="Qlib"
            title="Xếp hạng"
            trailing={
              scores?.date ? <span className="badge badge-info whitespace-nowrap">{scores.date}</span> : null
            }
          />
          {!scores || scores.rows.length === 0 ? (
            <div className="pixel-border bg-white p-5 text-[13px] font-mono text-[#555555]">
              Chưa có điểm Qlib. Bấm “Chạy Qlib” ở trên, hoặc chạy <code>python run.py qlib-refresh</code>.
            </div>
          ) : (
            <div className="pixel-border bg-white overflow-x-auto">
              <table className="w-full text-[13px] font-mono">
                <thead>
                  <tr className="border-b-2 border-black text-left">
                    <th className="px-4 py-2.5 w-14">#</th>
                    <th className="px-4 py-2.5">Mã</th>
                    <th className="px-4 py-2.5 text-right">Điểm mô hình</th>
                    <th className="px-4 py-2.5 w-[40%]">Xếp hạng trong rổ</th>
                  </tr>
                </thead>
                <tbody>
                  {scores.rows.map((r, i) => {
                    const pct = Math.round(r.rank_pct * 100);
                    return (
                      <tr key={r.ticker} className="border-b border-[#E8E2D6] last:border-0">
                        <td className="px-4 py-2 text-[#888888]">{i + 1}</td>
                        <td className="px-4 py-2 font-bold text-black">{r.ticker}</td>
                        <td className="px-4 py-2 text-right tabular-nums">{r.score.toFixed(4)}</td>
                        <td className="px-4 py-2">
                          <div className="flex items-center gap-3">
                            <div className="h-2.5 flex-1 bg-[#F3EFE7] border border-black">
                              <div
                                className={`h-full ${pct >= 70 ? "bg-[#10A37F]" : pct <= 30 ? "bg-[#EF4444]" : "bg-[#FAAD14]"}`}
                                style={{ width: `${pct}%` }}
                              />
                            </div>
                            <span className="w-10 text-right tabular-nums">{pct}</span>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              <p className="px-4 py-3 border-t-2 border-black text-[11px] text-[#555555] leading-snug">
                Điểm = dự báo lợi suất tương đối của LightGBM trên 158 factor Alpha158, chỉ dùng để so sánh các mã
                trong cùng ngày. Mô hình chưa thấy dữ liệu sau {run?.valid_end ?? "ngày huấn luyện"}. Rank IC dưới
                0,02 nghĩa là tín hiệu yếu — chỉ tham khảo, không phải khuyến nghị đầu tư.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
