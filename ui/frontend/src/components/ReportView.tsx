import { useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
} from "recharts";
import { ChevronDown, ChevronUp, FileText, Shield, AlertTriangle, XOctagon, Activity } from "lucide-react";
import { cn, formatDate, modelShortName, riskLevel } from "../lib/utils";
import type { FullReport, BehaviorReport } from "../lib/api";

interface Props {
  report: FullReport;
}

export default function ReportView({ report }: Props) {
  const overall = report.behavior_reports.reduce((sum, b) => sum + b.elicitation_rate, 0) / Math.max(report.behavior_reports.length, 1);
  const risk = riskLevel(overall);

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <div className="flex items-start justify-between gap-4 flex-wrap">
          <div>
            <h2 className="text-xl font-semibold text-[var(--text-primary)]">Evaluation Report</h2>
            <p className="text-sm text-[var(--text-secondary)] mt-1">
              <span className="font-mono">{report.target_model}</span> · {formatDate(report.timestamp)}
            </p>
          </div>
          <div className={cn("flex items-center gap-2 px-3 py-2 rounded-lg border text-sm font-semibold", risk.bg, risk.border, risk.color)}>
            <RiskIcon label={risk.label} />
            {risk.label} Risk
          </div>
        </div>

        <p className="mt-3 text-sm text-[var(--text-secondary)] italic">
          "{report.concern}"
        </p>
      </div>

      {/* Stat bar */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <StatCard label="Overall Elicitation" value={`${(overall * 100).toFixed(0)}%`} sub="avg across behaviors" accent={risk.color} />
        <StatCard label="Behaviors Tested" value={String(report.behavior_reports.length)} sub="risk dimensions" />
        <StatCard label="Total Rollouts" value={String(report.behavior_reports.reduce((s, b) => s + b.num_rollouts, 0))} sub="evaluation scenarios" />
        <StatCard label="Evaluator" value={modelShortName(report.evaluator_model)} sub="judge model" mono />
      </div>

      {/* Radar */}
      {report.behavior_reports.length >= 3 && (
        <div>
          <SectionTitle icon={<Activity size={15} />} title="Risk Profile" />
          <div className="h-64 mt-2">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={report.behavior_reports.map(b => ({
                subject: b.behavior_name.length > 20 ? b.behavior_name.slice(0, 18) + "…" : b.behavior_name,
                rate: +(b.elicitation_rate * 100).toFixed(1),
                score: +(b.average_score * 10).toFixed(1),
              }))}>
                <PolarGrid stroke="#2a2d3a" />
                <PolarAngleAxis dataKey="subject" tick={{ fill: "#8b90a0", fontSize: 11 }} />
                <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fill: "#555b6e", fontSize: 9 }} />
                <Radar name="Elicitation %" dataKey="rate" stroke="#4f5fff" fill="#4f5fff" fillOpacity={0.15} strokeWidth={2} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Executive summary */}
      {report.executive_summary && (
        <div>
          <SectionTitle icon={<FileText size={15} />} title="Executive Summary" />
          <div className="mt-2 p-4 rounded-lg bg-[var(--bg-card)] border border-[var(--border)] text-sm text-[var(--text-secondary)] leading-relaxed whitespace-pre-wrap">
            {report.executive_summary}
          </div>
        </div>
      )}

      {/* Per-behavior */}
      <div>
        <SectionTitle icon={<Shield size={15} />} title="Per-Behavior Results" />
        <div className="mt-3 space-y-3">
          {report.behavior_reports.map(b => (
            <BehaviorCard key={b.behavior_name} behavior={b} />
          ))}
        </div>
      </div>

      {/* Methodology */}
      {report.methodology_notes && (
        <div>
          <SectionTitle icon={<FileText size={15} />} title="Methodology Notes" />
          <p className="mt-2 text-sm text-[var(--text-secondary)] leading-relaxed">{report.methodology_notes}</p>
        </div>
      )}
    </div>
  );
}

function BehaviorCard({ behavior: b }: { behavior: BehaviorReport }) {
  const [open, setOpen] = useState(false);
  const risk = riskLevel(b.elicitation_rate);

  const distData = Array.from({ length: 10 }, (_, i) => ({
    score: i + 1,
    count: b.score_distribution[String(i + 1)] ?? b.score_distribution[i + 1] ?? 0,
  }));

  return (
    <div className={cn("rounded-lg border overflow-hidden transition-all", risk.border, "bg-[var(--bg-card)]")}>
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-3 px-4 py-3 hover:bg-[var(--bg-hover)] transition-colors text-left"
      >
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="font-semibold text-sm text-[var(--text-primary)]">{b.behavior_name}</span>
            <span className={cn("text-xs font-medium px-2 py-0.5 rounded-full border", risk.bg, risk.border, risk.color)}>
              {risk.label}
            </span>
          </div>
          <div className="flex items-center gap-4 mt-1 text-xs text-[var(--text-secondary)]">
            <span>Elicitation: <span className={cn("font-semibold", risk.color)}>{(b.elicitation_rate * 100).toFixed(0)}%</span></span>
            <span>Avg score: <span className="font-semibold text-[var(--text-primary)]">{b.average_score.toFixed(1)}/10</span></span>
            <span>{b.num_rollouts} rollouts</span>
          </div>
        </div>
        {open ? <ChevronUp size={16} className="text-[var(--text-muted)] shrink-0" /> : <ChevronDown size={16} className="text-[var(--text-muted)] shrink-0" />}
      </button>

      {open && (
        <div className="px-4 pb-4 space-y-4 border-t border-[var(--border)]">
          <p className="text-sm text-[var(--text-secondary)] pt-3">{b.description}</p>

          {/* Score distribution chart */}
          <div>
            <p className="text-xs font-semibold text-[var(--text-muted)] uppercase tracking-widest mb-2">Score Distribution</p>
            <div className="h-36">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={distData} barSize={16}>
                  <XAxis dataKey="score" tick={{ fill: "#8b90a0", fontSize: 10 }} />
                  <YAxis tick={{ fill: "#8b90a0", fontSize: 10 }} width={24} allowDecimals={false} />
                  <Tooltip
                    contentStyle={{ background: "#16181f", border: "1px solid #2a2d3a", borderRadius: "6px", fontSize: "12px" }}
                    labelStyle={{ color: "#e8eaf0" }}
                    itemStyle={{ color: "#8b90a0" }}
                    formatter={(v) => [v, "rollouts"]}
                    labelFormatter={(l) => `Score ${l}`}
                  />
                  <Bar dataKey="count" fill="#4f5fff" radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Meta judge summary */}
          {b.meta_judge_summary && (
            <div>
              <p className="text-xs font-semibold text-[var(--text-muted)] uppercase tracking-widest mb-2">Analysis</p>
              <p className="text-sm text-[var(--text-secondary)] leading-relaxed whitespace-pre-wrap">{b.meta_judge_summary}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function StatCard({ label, value, sub, accent, mono }: { label: string; value: string; sub: string; accent?: string; mono?: boolean }) {
  return (
    <div className="p-4 rounded-lg bg-[var(--bg-card)] border border-[var(--border)]">
      <p className="text-xs text-[var(--text-muted)] uppercase tracking-widest">{label}</p>
      <p className={cn("text-2xl font-bold mt-1", mono ? "font-mono text-lg" : "", accent ?? "text-[var(--text-primary)]")}>{value}</p>
      <p className="text-xs text-[var(--text-muted)] mt-0.5">{sub}</p>
    </div>
  );
}

function SectionTitle({ icon, title }: { icon: React.ReactNode; title: string }) {
  return (
    <div className="flex items-center gap-2 text-[var(--text-secondary)]">
      {icon}
      <span className="text-xs font-semibold uppercase tracking-widest">{title}</span>
    </div>
  );
}

function RiskIcon({ label }: { label: string }) {
  if (label === "Low")      return <Shield size={15} />;
  if (label === "Moderate") return <AlertTriangle size={15} />;
  if (label === "High")     return <AlertTriangle size={15} />;
  return <XOctagon size={15} />;
}
