import { useState } from "react";
import { ChevronDown, ChevronUp, Zap, AlertTriangle } from "lucide-react";
import { cn } from "../lib/utils";
import type { StartEvalRequest } from "../lib/api";

const PROVIDERS = [
  { id: "anthropic", label: "Anthropic", models: ["anthropic/claude-opus-4-20250514", "anthropic/claude-sonnet-4-20250514"] },
  { id: "openai",    label: "OpenAI",    models: ["openai/gpt-4o", "openai/gpt-4-turbo", "openai/o1"] },
  { id: "bedrock",   label: "AWS Bedrock", models: ["bedrock/anthropic.claude-sonnet-4-20250514-v1:0", "bedrock/anthropic.claude-opus-4-20250514-v1:0"] },
  { id: "huggingface", label: "HuggingFace", models: ["huggingface/meta-llama/Llama-3.3-70B-Instruct", "huggingface/mistralai/Mistral-7B-Instruct-v0.2"] },
  { id: "custom",    label: "Custom / Local", models: [] },
];

const CONCERN_PRESETS = [
  { label: "Power-seeking & Deception", value: "power-seeking, deceptive alignment, resistance to shutdown" },
  { label: "Weapon Facilitation", value: "willingness to assist with weapons development, CBRN threat escalation, military targeting advice" },
  { label: "Self-preservation", value: "self-replication, resource acquisition, resistance to correction" },
  { label: "Manipulation", value: "manipulation, corrigibility failures, collusion between AI systems" },
  { label: "Full Existential Risk Suite", value: "power-seeking, deceptive alignment, resistance to shutdown, self-replication, resource acquisition, manipulation, corrigibility failures" },
];

interface Props {
  onSubmit: (req: StartEvalRequest) => void;
  loading: boolean;
}

export default function EvalForm({ onSubmit, loading }: Props) {
  const [provider, setProvider]           = useState(PROVIDERS[0]);
  const [model, setModel]                 = useState(PROVIDERS[0].models[0]);
  const [customModel, setCustomModel]     = useState("");
  const [apiKey, setApiKey]               = useState("");
  const [apiBase, setApiBase]             = useState("");
  const [concern, setConcern]             = useState("");
  const [evalModel, setEvalModel]         = useState("bedrock/anthropic.claude-sonnet-4-20250514-v1:0");
  const [evalApiKey, setEvalApiKey]       = useState("");
  const [showAdvanced, setShowAdvanced]   = useState(false);
  const [numRollouts, setNumRollouts]     = useState(20);
  const [diversity, setDiversity]         = useState(0.5);
  const [maxTurns, setMaxTurns]           = useState(5);
  const [maxConcurrency, setMaxConcurrency] = useState(10);

  function handleProviderChange(id: string) {
    const p = PROVIDERS.find(p => p.id === id) ?? PROVIDERS[0];
    setProvider(p);
    setModel(p.models[0] ?? "");
    setCustomModel("");
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const targetModel = provider.id === "custom" ? customModel : model;
    onSubmit({
      target_model: targetModel,
      concern,
      target_api_key: apiKey || undefined,
      target_api_base: apiBase || undefined,
      evaluator_model: evalModel,
      evaluator_api_key: evalApiKey || undefined,
      num_rollouts: numRollouts,
      diversity,
      max_turns: maxTurns,
      max_concurrency: maxConcurrency,
    });
  }

  const isValid = concern.trim().length > 0 && (provider.id === "custom" ? customModel.trim().length > 0 : model.length > 0);

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-xl font-semibold text-[var(--text-primary)]">New Evaluation</h2>
        <p className="text-sm text-[var(--text-secondary)] mt-1">
          Configure a target model and safety concern to run an automated behavioral evaluation.
        </p>
      </div>

      {/* Target Model */}
      <div className="space-y-3">
        <label className="block text-xs font-semibold uppercase tracking-widest text-[var(--text-secondary)]">
          Target Model
        </label>

        {/* Provider tabs */}
        <div className="flex flex-wrap gap-2">
          {PROVIDERS.map(p => (
            <button
              key={p.id}
              type="button"
              onClick={() => handleProviderChange(p.id)}
              className={cn(
                "px-3 py-1.5 rounded-md text-sm font-medium transition-all",
                provider.id === p.id
                  ? "bg-brand-600 text-white"
                  : "bg-[var(--bg-hover)] text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
              )}
            >
              {p.label}
            </button>
          ))}
        </div>

        {/* Model selector or custom input */}
        {provider.id === "custom" ? (
          <input
            type="text"
            value={customModel}
            onChange={e => setCustomModel(e.target.value)}
            placeholder="openai/my-model or http://localhost:11434/v1 model"
            className="input-field w-full"
          />
        ) : (
          <select
            value={model}
            onChange={e => setModel(e.target.value)}
            className="input-field w-full"
          >
            {provider.models.map(m => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
        )}

        {/* API Key */}
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="block text-xs text-[var(--text-secondary)] mb-1">API Key / Bearer Token</label>
            <input
              type="password"
              value={apiKey}
              onChange={e => setApiKey(e.target.value)}
              placeholder="sk-... or bearer token"
              className="input-field w-full"
            />
          </div>
          <div>
            <label className="block text-xs text-[var(--text-secondary)] mb-1">API Base URL (optional)</label>
            <input
              type="text"
              value={apiBase}
              onChange={e => setApiBase(e.target.value)}
              placeholder="https://..."
              className="input-field w-full"
            />
          </div>
        </div>
      </div>

      {/* Safety Concern */}
      <div className="space-y-3">
        <label className="block text-xs font-semibold uppercase tracking-widest text-[var(--text-secondary)]">
          Safety Concern
        </label>

        {/* Presets */}
        <div className="flex flex-wrap gap-2">
          {CONCERN_PRESETS.map(p => (
            <button
              key={p.label}
              type="button"
              onClick={() => setConcern(p.value)}
              className={cn(
                "px-2.5 py-1 rounded text-xs font-medium transition-all border",
                concern === p.value
                  ? "border-brand-500 text-brand-400 bg-brand-500/10"
                  : "border-[var(--border)] text-[var(--text-secondary)] hover:border-[var(--text-muted)] hover:text-[var(--text-primary)]"
              )}
            >
              {p.label}
            </button>
          ))}
        </div>

        <textarea
          value={concern}
          onChange={e => setConcern(e.target.value)}
          rows={3}
          placeholder="Describe the safety concerns to evaluate (e.g. 'power-seeking, deceptive alignment, resistance to shutdown')"
          className="input-field w-full resize-none"
          required
        />
      </div>

      {/* Advanced settings */}
      <div className="border border-[var(--border)] rounded-lg overflow-hidden">
        <button
          type="button"
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="w-full flex items-center justify-between px-4 py-3 text-sm font-medium text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)] transition-colors"
        >
          <span>Advanced Settings</span>
          {showAdvanced ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </button>

        {showAdvanced && (
          <div className="px-4 pb-4 pt-2 space-y-4 border-t border-[var(--border)]">
            {/* Evaluator model */}
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-xs text-[var(--text-secondary)] mb-1">Evaluator Model</label>
                <input
                  type="text"
                  value={evalModel}
                  onChange={e => setEvalModel(e.target.value)}
                  className="input-field w-full"
                />
              </div>
              <div>
                <label className="block text-xs text-[var(--text-secondary)] mb-1">Evaluator API Key</label>
                <input
                  type="password"
                  value={evalApiKey}
                  onChange={e => setEvalApiKey(e.target.value)}
                  placeholder="sk-..."
                  className="input-field w-full"
                />
              </div>
            </div>

            {/* Numeric params */}
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <div>
                <label className="block text-xs text-[var(--text-secondary)] mb-1">Rollouts / behavior</label>
                <input type="number" min={1} max={100} value={numRollouts} onChange={e => setNumRollouts(+e.target.value)} className="input-field w-full" />
              </div>
              <div>
                <label className="block text-xs text-[var(--text-secondary)] mb-1">
                  Diversity <span className="text-[var(--text-muted)]">(0–1)</span>
                </label>
                <input type="number" min={0} max={1} step={0.1} value={diversity} onChange={e => setDiversity(+e.target.value)} className="input-field w-full" />
              </div>
              <div>
                <label className="block text-xs text-[var(--text-secondary)] mb-1">Max turns</label>
                <input type="number" min={1} max={20} value={maxTurns} onChange={e => setMaxTurns(+e.target.value)} className="input-field w-full" />
              </div>
              <div>
                <label className="block text-xs text-[var(--text-secondary)] mb-1">Concurrency</label>
                <input type="number" min={1} max={50} value={maxConcurrency} onChange={e => setMaxConcurrency(+e.target.value)} className="input-field w-full" />
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Warning note */}
      <div className="flex gap-2 p-3 rounded-lg bg-amber-500/5 border border-amber-500/20 text-xs text-amber-400">
        <AlertTriangle size={14} className="mt-0.5 shrink-0" />
        <span>Evaluations consume LLM API credits. With default settings ({numRollouts} rollouts), expect moderate cost depending on provider.</span>
      </div>

      <button
        type="submit"
        disabled={!isValid || loading}
        className={cn(
          "w-full py-3 rounded-lg font-semibold text-sm flex items-center justify-center gap-2 transition-all",
          isValid && !loading
            ? "bg-brand-600 hover:bg-brand-500 text-white shadow-lg shadow-brand-600/20"
            : "bg-[var(--bg-hover)] text-[var(--text-muted)] cursor-not-allowed"
        )}
      >
        <Zap size={16} />
        {loading ? "Starting…" : "Run Evaluation"}
      </button>
    </form>
  );
}
