import { useEffect, useState } from "react";
import { Trash2, RefreshCw, Plus, Clock, CheckCircle2, XCircle, Loader } from "lucide-react";
import { cn, formatDate, modelShortName } from "../lib/utils";
import { deleteEvaluation, listEvaluations, type EvalListItem } from "../lib/api";

interface Props {
  selectedId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  refreshSignal: number;
}

export default function HistorySidebar({ selectedId, onSelect, onNew, refreshSignal }: Props) {
  const [items, setItems]     = useState<EvalListItem[]>([]);
  const [loading, setLoading] = useState(false);

  async function load() {
    setLoading(true);
    try {
      setItems(await listEvaluations());
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { load(); }, [refreshSignal]);

  async function handleDelete(e: React.MouseEvent, id: string) {
    e.stopPropagation();
    await deleteEvaluation(id);
    setItems(prev => prev.filter(i => i.run_id !== id));
    if (selectedId === id) onNew();
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-[var(--border)]">
        <span className="text-xs font-semibold uppercase tracking-widest text-[var(--text-secondary)]">Evaluations</span>
        <div className="flex items-center gap-1">
          <button onClick={load} className="p-1.5 rounded hover:bg-[var(--bg-hover)] text-[var(--text-muted)] hover:text-[var(--text-secondary)] transition-colors">
            <RefreshCw size={13} className={cn(loading && "animate-spin")} />
          </button>
          <button onClick={onNew} className="p-1.5 rounded hover:bg-[var(--bg-hover)] text-[var(--text-muted)] hover:text-[var(--text-secondary)] transition-colors">
            <Plus size={13} />
          </button>
        </div>
      </div>

      {/* List */}
      <div className="flex-1 overflow-y-auto">
        {items.length === 0 && !loading && (
          <div className="px-4 py-8 text-center text-xs text-[var(--text-muted)]">
            No evaluations yet
          </div>
        )}

        {items.map(item => (
          <SidebarItem
            key={item.run_id}
            item={item}
            selected={selectedId === item.run_id}
            onSelect={() => onSelect(item.run_id)}
            onDelete={e => handleDelete(e, item.run_id)}
          />
        ))}
      </div>
    </div>
  );
}

function SidebarItem({
  item, selected, onSelect, onDelete,
}: {
  item: EvalListItem;
  selected: boolean;
  onSelect: () => void;
  onDelete: (e: React.MouseEvent) => void;
}) {
  return (
    <div
      role="button"
      tabIndex={0}
      onClick={onSelect}
      onKeyDown={e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onSelect(); } }}
      className={cn(
        "w-full text-left px-4 py-3 border-b border-[var(--border-subtle)] hover:bg-[var(--bg-hover)] transition-colors group relative cursor-pointer",
        selected && "bg-brand-500/8 border-l-2 border-l-brand-500"
      )}
    >
      <div className="flex items-start gap-2">
        <StatusIcon status={item.status} />
        <div className="flex-1 min-w-0">
          <p className="text-xs font-medium text-[var(--text-primary)] truncate">
            {modelShortName(item.target_model)}
          </p>
          <p className="text-xs text-[var(--text-secondary)] truncate mt-0.5 leading-snug">
            {item.concern}
          </p>
          <p className="text-xs text-[var(--text-muted)] mt-1 flex items-center gap-1">
            <Clock size={10} />
            {formatDate(item.created_at)}
          </p>
        </div>
        <button
          type="button"
          onClick={onDelete}
          className="opacity-0 group-hover:opacity-100 p-1 rounded text-[var(--text-muted)] hover:text-red-400 transition-all shrink-0"
        >
          <Trash2 size={12} />
        </button>
      </div>
    </div>
  );
}

function StatusIcon({ status }: { status: string }) {
  if (status === "completed") return <CheckCircle2 size={12} className="text-risk-low mt-0.5 shrink-0" />;
  if (status === "failed")    return <XCircle size={12} className="text-risk-high mt-0.5 shrink-0" />;
  return <Loader size={12} className="text-brand-400 animate-spin mt-0.5 shrink-0" />;
}
