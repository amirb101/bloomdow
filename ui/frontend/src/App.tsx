import { useState } from "react";
import { Shield } from "lucide-react";
import EvalForm from "./components/EvalForm";
import LiveProgress from "./components/LiveProgress";
import ReportView from "./components/ReportView";
import HistorySidebar from "./components/HistorySidebar";
import { startEvaluation, streamProgress, getEvaluation, type StartEvalRequest, type StageEvent, type FullReport } from "./lib/api";

type View = "form" | "progress" | "report";

export default function App() {
  const [view, setView]           = useState<View>("form");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [stages, setStages]       = useState<Record<number, StageEvent>>({});
  const [report, setReport]       = useState<FullReport | null>(null);
  const [progressError, setProgressError] = useState<string | undefined>();
  const [formLoading, setFormLoading] = useState(false);
  const [historyRefresh, setHistoryRefresh] = useState(0);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  async function handleStartEval(req: StartEvalRequest) {
    setFormLoading(true);
    try {
      const { run_id } = await startEvaluation(req);
      setSelectedId(run_id);
      setStages({});
      setReport(null);
      setProgressError(undefined);
      setView("progress");
      setHistoryRefresh(n => n + 1);

      streamProgress(
        run_id,
        (stage) => setStages(prev => ({ ...prev, [stage.stage]: stage })),
        (complete) => {
          setReport(complete.report);
          setView("report");
          setHistoryRefresh(n => n + 1);
        },
        (err) => {
          setProgressError(err);
          setHistoryRefresh(n => n + 1);
        },
      );
    } catch (err: any) {
      alert("Failed to start evaluation: " + err.message);
    } finally {
      setFormLoading(false);
    }
  }

  async function handleSelectHistory(runId: string) {
    setSelectedId(runId);
    const record = await getEvaluation(runId);
    if (record.status === "completed" && record.report) {
      setReport(record.report);
      setView("report");
    } else if (record.status === "running") {
      setStages({});
      setReport(null);
      setProgressError(undefined);
      setView("progress");
      streamProgress(
        runId,
        (stage) => setStages(prev => ({ ...prev, [stage.stage]: stage })),
        (complete) => {
          setReport(complete.report);
          setView("report");
          setHistoryRefresh(n => n + 1);
        },
        (err) => {
          setProgressError(err);
          setHistoryRefresh(n => n + 1);
        },
      );
    } else {
      setView("form");
    }
  }

  function handleNew() {
    setSelectedId(null);
    setStages({});
    setReport(null);
    setProgressError(undefined);
    setView("form");
  }

  return (
    <div className="flex h-screen bg-[var(--bg-primary)] overflow-hidden">
      {/* Sidebar */}
      <div className={`${sidebarOpen ? "w-72" : "w-0 overflow-hidden"} shrink-0 flex flex-col bg-[var(--bg-secondary)] border-r border-[var(--border)] transition-all duration-200`}>
        {/* Logo */}
        <div className="flex items-center gap-2.5 px-4 py-4 border-b border-[var(--border)]">
          <div className="w-7 h-7 rounded-lg bg-brand-600 flex items-center justify-center">
            <Shield size={14} className="text-white" />
          </div>
          <div>
            <p className="text-sm font-bold text-[var(--text-primary)] leading-none">Bloomdow</p>
            <p className="text-xs text-[var(--text-muted)] mt-0.5">AI Safety Evals</p>
          </div>
        </div>

        {/* New Eval button */}
        <div className="px-4 py-3 border-b border-[var(--border)]">
          <button
            onClick={handleNew}
            className="w-full py-2 rounded-lg bg-brand-600 hover:bg-brand-500 text-white text-sm font-semibold transition-colors"
          >
            + New Evaluation
          </button>
        </div>

        <HistorySidebar
          selectedId={selectedId}
          onSelect={handleSelectHistory}
          onNew={handleNew}
          refreshSignal={historyRefresh}
        />
      </div>

      {/* Sidebar toggle */}
      <button
        onClick={() => setSidebarOpen(!sidebarOpen)}
        className="absolute left-0 top-1/2 -translate-y-1/2 z-10 w-4 h-8 bg-[var(--bg-card)] border border-[var(--border)] rounded-r flex items-center justify-center text-[var(--text-muted)] hover:text-[var(--text-secondary)] transition-all"
        style={{ left: sidebarOpen ? "288px" : "0" }}
      >
        <span className="text-xs">{sidebarOpen ? "‹" : "›"}</span>
      </button>

      {/* Main content */}
      <main className="flex-1 overflow-y-auto">
        <div className="max-w-3xl mx-auto px-6 py-8">
          {view === "form" && (
            <EvalForm onSubmit={handleStartEval} loading={formLoading} />
          )}
          {view === "progress" && selectedId && (
            <LiveProgress stages={stages} error={progressError} runId={selectedId} />
          )}
          {view === "report" && report && (
            <ReportView report={report} />
          )}
        </div>
      </main>
    </div>
  );
}
