import { CheckCircle, Circle, Loader, XCircle } from "lucide-react";
import { cn } from "../lib/utils";
import type { StageEvent } from "../lib/api";

const STAGES = [
  { num: 1, name: "Scoping",      desc: "Decomposing concern into behavioral dimensions" },
  { num: 2, name: "Understanding", desc: "Deep analysis of each risk dimension" },
  { num: 3, name: "Ideation",     desc: "Generating evaluation scenarios" },
  { num: 4, name: "Rollout",      desc: "Executing scenarios against target model" },
  { num: 5, name: "Judgment",     desc: "Scoring transcripts and generating report" },
];

interface Props {
  stages: Record<number, StageEvent>;
  error?: string;
  runId: string;
}

export default function LiveProgress({ stages, error, runId }: Props) {
  const behaviors = stages[1]?.behaviors;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-[var(--text-primary)]">Evaluation Running</h2>
        <p className="text-sm text-[var(--text-secondary)] mt-1 font-mono">run/{runId}</p>
      </div>

      {/* Stage progress */}
      <div className="space-y-2">
        {STAGES.map(stage => {
          const ev = stages[stage.num];
          const isComplete = ev?.status === "complete";
          const isRunning  = ev?.status === "running";
          const isPending  = !ev;
          const isFailed   = error && isRunning;

          return (
            <div
              key={stage.num}
              className={cn(
                "flex items-start gap-3 p-3 rounded-lg border transition-all",
                isComplete && "border-brand-500/30 bg-brand-500/5",
                isRunning  && "border-brand-400/60 bg-brand-500/10",
                isPending  && "border-[var(--border-subtle)] bg-transparent opacity-40",
                isFailed   && "border-red-500/40 bg-red-500/5",
              )}
            >
              {/* Icon */}
              <div className="mt-0.5">
                {isFailed   ? <XCircle size={18} className="text-red-400" /> :
                 isComplete ? <CheckCircle size={18} className="text-brand-400" /> :
                 isRunning  ? <Loader size={18} className="text-brand-400 animate-spin" /> :
                              <Circle size={18} className="text-[var(--text-muted)]" />}
              </div>

              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-mono text-[var(--text-muted)]">{stage.num}/5</span>
                  <span className={cn("text-sm font-semibold", isComplete || isRunning ? "text-[var(--text-primary)]" : "text-[var(--text-secondary)]")}>
                    {stage.name}
                  </span>
                  {isComplete && (
                    <span className="ml-auto text-xs text-brand-400 font-medium">Done</span>
                  )}
                </div>
                <p className="text-xs text-[var(--text-secondary)] mt-0.5">
                  {ev?.detail ?? stage.desc}
                </p>
              </div>
            </div>
          );
        })}
      </div>

      {/* Behaviors found after scoping */}
      {behaviors && behaviors.length > 0 && (
        <div>
          <p className="text-xs font-semibold uppercase tracking-widest text-[var(--text-secondary)] mb-2">
            Identified Risk Dimensions
          </p>
          <div className="space-y-1.5">
            {behaviors.map(b => (
              <div key={b.name} className="flex items-start gap-2 p-2 rounded bg-[var(--bg-card)]">
                <span className="inline-block w-1.5 h-1.5 rounded-full bg-brand-400 mt-1.5 shrink-0" />
                <div>
                  <span className="text-sm font-medium text-[var(--text-primary)]">{b.name}</span>
                  <span className="text-xs text-[var(--text-muted)] ml-2">{b.modality}</span>
                  <p className="text-xs text-[var(--text-secondary)] mt-0.5">{b.description}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {error && (
        <div className="p-4 rounded-lg bg-red-500/10 border border-red-500/30 text-sm text-red-400">
          <p className="font-semibold mb-1">Evaluation failed</p>
          <p className="font-mono text-xs break-all">{error}</p>
        </div>
      )}
    </div>
  );
}
