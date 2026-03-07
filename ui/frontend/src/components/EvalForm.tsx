import { useState } from "react";
import { ChevronDown, ChevronUp, Zap, AlertTriangle } from "lucide-react";
import { cn } from "../lib/utils";
import type { StartEvalRequest } from "../lib/api";

const PROVIDERS = [
  { id: "anthropic", label: "Anthropic", models: ["anthropic/claude-opus-4-20250514", "anthropic/claude-sonnet-4-20250514"], keyHint: "sk-ant-..." },
  { id: "openai",    label: "OpenAI",    models: ["openai/gpt-4o", "openai/gpt-4-turbo", "openai/o1"], keyHint: "sk-..." },
  { id: "deepseek",  label: "DeepSeek",  models: ["deepseek/deepseek-chat", "deepseek/deepseek-coder", "deepseek/deepseek-reasoner"], keyHint: "sk-... (from platform.deepseek.com)" },
  { id: "bedrock",   label: "AWS Bedrock", models: ["bedrock/anthropic.claude-sonnet-4-20250514-v1:0", "bedrock/anthropic.claude-opus-4-20250514-v1:0"], keyHint: "Bearer token" },
  { id: "huggingface", label: "HuggingFace", models: ["huggingface/meta-llama/Llama-3.3-70B-Instruct", "huggingface/mistralai/Mistral-7B-Instruct-v0.2", "huggingface/together/deepseek-ai/DeepSeek-R1"], keyHint: "hf_... (from huggingface.co/settings/tokens)" },
  { id: "custom",    label: "Custom / Local", models: [], keyHint: "API key for your endpoint" },
];

const EVALUATOR_PRESETS = [
  { id: "deepseek", label: "DeepSeek (cheap)", model: "deepseek/deepseek-chat", keyHint: "DeepSeek API key" },
  { id: "gpt4o-mini", label: "GPT-4o-mini (cheap)", model: "openai/gpt-4o-mini", keyHint: "OpenAI API key" },
  { id: "claude-haiku", label: "Claude Haiku", model: "anthropic/claude-3-5-haiku-20241022", keyHint: "Anthropic API key" },
  { id: "bedrock", label: "Bedrock (default)", model: "bedrock/anthropic.claude-sonnet-4-20250514-v1:0", keyHint: "AWS bearer token" },
  { id: "custom", label: "Custom", model: "", keyHint: "Enter model below" },
];

const COST_PRESETS = [
  { id: "low",    label: "Low",    rollouts: 2,  maxTurns: 3, desc: "~2 rollouts/behavior, quick smoke test" },
  { id: "medium", label: "Medium", rollouts: 10, maxTurns: 5, desc: "~10 rollouts, balanced" },
  { id: "high",   label: "High",   rollouts: 20, maxTurns: 5, desc: "~20 rollouts, thorough" },
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
  const [useSavedKeys, setUseSavedKeys]   = useState(false);
  const [apiKey, setApiKey]               = useState("");
  const [apiBase, setApiBase]             = useState("");
  const [concern, setConcern]             = useState("");
  const [evalPreset, setEvalPreset]       = useState("deepseek");
  const [evalModel, setEvalModel]         = useState("deepseek/deepseek-chat");
  const [evalApiKey, setEvalApiKey]       = useState("");
  const [showAdvanced, setShowAdvanced]   = useState(false);
  const [costPreset, setCostPreset]       = useState<"low" | "medium" | "high">("low");
  const [numRollouts, setNumRollouts]     = useState(2);
  const [maxTurns, setMaxTurns]           = useState(3);
  const [maxConcurrency, setMaxConcurrency] = useState(3);

  function applyEvalPreset(id: string) {
    const p = EVALUATOR_PRESETS.find(x => x.id === id);
    if (!p) return;
    setEvalPreset(id);
    if (p.model) setEvalModel(p.model);
  }

  function applyCostPreset(id: "low" | "medium" | "high") {
    const p = COST_PRESETS.find(x => x.id === id)!;
    setCostPreset(id);
    setNumRollouts(p.rollouts);
    setMaxTurns(p.maxTurns);
  }

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
      target_api_key: useSavedKeys ? undefined : (apiKey || undefined),
      target_api_base: apiBase || undefined,
      evaluator_model: evalModel,
      evaluator_api_key: useSavedKeys ? undefined : (evalApiKey || undefined),
      num_rollouts: numRollouts,
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

      {/* Config — saved keys */}
      <div className="border border-[var(--border)] rounded-lg p-4 bg-[var(--bg-card)]">
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={useSavedKeys}
            onChange={e => setUseSavedKeys(e.target.checked)}
            className="rounded border-[var(--border)] bg-[var(--bg-primary)] text-brand-600 focus:ring-brand-500"
          />
          <span className="text-sm font-medium text-[var(--text-primary)]">Use saved keys (from server config)</span>
        </label>
        <p className="text-xs text-[var(--text-muted)] mt-2">
          Keys are read from <code className="bg-[var(--bg-hover)] px-1 rounded">ui/backend/.env</code>. Copy from <code className="bg-[var(--bg-hover)] px-1 rounded">.env.example</code> and fill in. Never stored in the browser.
        </p>
      </div>

      {/* Target Model — Step 1: Provider */}
      <div className="space-y-3">
        <label className="block text-xs font-semibold uppercase tracking-widest text-[var(--text-secondary)]">
          1. Select provider & model
        </label>

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

        {provider.id === "custom" ? (
          <input
            type="text"
            value={customModel}
            onChange={e => setCustomModel(e.target.value)}
            placeholder="e.g. openai/gpt-4o or openai/model-name with custom api_base"
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

        {/* Step 2: API Key — hidden when using saved keys */}
        {!useSavedKeys && (
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs text-[var(--text-secondary)] mb-1">
                2. API key for {provider.label}
              </label>
              <input
                type="password"
                value={apiKey}
                onChange={e => setApiKey(e.target.value)}
                placeholder={provider.keyHint}
                className="input-field w-full"
              />
            </div>
            <div>
              <label className="block text-xs text-[var(--text-secondary)] mb-1">API base URL <span className="text-[var(--text-muted)]">(optional)</span></label>
              <input
                type="text"
                value={apiBase}
                onChange={e => setApiBase(e.target.value)}
                placeholder="Custom endpoint, e.g. https://..."
                className="input-field w-full"
              />
            </div>
          </div>
        )}
        {useSavedKeys && (
          <p className="text-xs text-[var(--text-muted)]">Using keys from <code className="bg-[var(--bg-hover)] px-1 rounded">.env</code></p>
        )}
      </div>

      {/* Safety Concern */}
      <div className="space-y-3">
        <label className="block text-xs font-semibold uppercase tracking-widest text-[var(--text-secondary)]">
          3. Safety concern to evaluate
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
            {/* Cost presets */}
            <div>
              <label className="block text-xs text-[var(--text-secondary)] mb-2">Cost preset (queries / rollouts)</label>
              <div className="flex flex-wrap gap-2">
                {COST_PRESETS.map(p => (
                  <button
                    key={p.id}
                    type="button"
                    onClick={() => applyCostPreset(p.id as "low" | "medium" | "high")}
                    className={cn(
                      "px-3 py-2 rounded-lg text-left border transition-all",
                      costPreset === p.id
                        ? "border-brand-500 bg-brand-500/10 text-brand-400"
                        : "border-[var(--border)] text-[var(--text-secondary)] hover:border-[var(--text-muted)] hover:text-[var(--text-primary)]"
                    )}
                  >
                    <span className="block text-sm font-medium">{p.label}</span>
                    <span className="block text-xs text-[var(--text-muted)] mt-0.5">{p.desc}</span>
                  </button>
                ))}
              </div>
            </div>

            {/* Evaluator model — runs the pipeline (scoping, ideation, judgment) */}
            <div>
              <label className="block text-xs text-[var(--text-secondary)] mb-2">Evaluator model (LiteLLM)</label>
              <div className="flex flex-wrap gap-2 mb-3">
                {EVALUATOR_PRESETS.map(p => (
                  <button
                    key={p.id}
                    type="button"
                    onClick={() => applyEvalPreset(p.id)}
                    className={cn(
                      "px-2.5 py-1.5 rounded text-sm font-medium transition-all border",
                      evalPreset === p.id
                        ? "border-brand-500 text-brand-400 bg-brand-500/10"
                        : "border-[var(--border)] text-[var(--text-secondary)] hover:border-[var(--text-muted)]"
                    )}
                  >
                    {p.label}
                  </button>
                ))}
              </div>
              <div className={cn("grid gap-3", useSavedKeys ? "grid-cols-1" : "grid-cols-2")}>
                <div>
                  <label className="block text-xs text-[var(--text-secondary)] mb-1">Model string</label>
                  <input
                    type="text"
                    value={evalModel}
                    onChange={e => { setEvalModel(e.target.value); setEvalPreset("custom"); }}
                    placeholder="provider/model-name"
                    className="input-field w-full"
                  />
                </div>
                {!useSavedKeys && (
                  <div>
                    <label className="block text-xs text-[var(--text-secondary)] mb-1">Evaluator API key</label>
                    <input
                      type="password"
                      value={evalApiKey}
                      onChange={e => setEvalApiKey(e.target.value)}
                      placeholder={EVALUATOR_PRESETS.find(p => p.id === evalPreset)?.keyHint ?? "sk-..."}
                      className="input-field w-full"
                    />
                  </div>
                )}
              </div>
            </div>

            {/* Numeric params */}
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
              <div>
                <label className="block text-xs text-[var(--text-secondary)] mb-1">Rollouts / behavior</label>
                <input type="number" min={1} max={100} value={numRollouts} onChange={e => setNumRollouts(+e.target.value)} className="input-field w-full" />
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
