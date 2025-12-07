import { type ChangeEvent, useMemo, useState } from "react";

type Prediction = {
  label: "authentic" | "fake";
  probabilities: { authentic: number; fake: number };
  confidence: number;
  feature_importances: { feature: string; importance: number; value: number }[];
};

type ApiResponse = {
  prediction: Prediction;
  features: Record<string, number>;
  feature_columns: string[];
  gestalt_status: string;
  model_version: string;
  metadata: {
    width?: number;
    height?: number;
    format?: string;
    file_size_bytes?: number;
  };
  file_name?: string;
};

type HistoryItem = {
  id: string;
  label: "authentic" | "fake";
  confidence: number;
  preview: string | null;
  timestamp: string;
  fileName?: string;
};

const API_BASE_DEFAULT = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const numberFmt = new Intl.NumberFormat("ja-JP", { maximumFractionDigits: 3 });

const byteFmt = (bytes?: number) => {
  if (!bytes && bytes !== 0) return "-";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
};

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [includeGestalt, setIncludeGestalt] = useState(false);
  const [apiBase, setApiBase] = useState<string>(API_BASE_DEFAULT);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ApiResponse | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);

  const handleFile = (inputFile: File | null) => {
    if (!inputFile) return;
    setFile(inputFile);
    const url = URL.createObjectURL(inputFile);
    setPreviewUrl(url);
  };

  const onFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0];
    if (selected) handleFile(selected);
  };

  const submit = async () => {
    if (!file) {
      setError("画像ファイルを選択してください。");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await fetch(`${apiBase}/predict?include_gestalt=${includeGestalt}`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || "推論リクエストに失敗しました。");
      }
      const data: ApiResponse = await res.json();
      setResult(data);
      const item: HistoryItem = {
        id: crypto.randomUUID(),
        label: data.prediction.label,
        confidence: data.prediction.confidence,
        preview: previewUrl,
        fileName: data.file_name,
        timestamp: new Date().toLocaleTimeString(),
      };
      setHistory((prev) => [item, ...prev].slice(0, 8));
    } catch (e) {
      setError(e instanceof Error ? e.message : "不明なエラーが発生しました。");
    } finally {
      setLoading(false);
    }
  };

  const colorFeatures = useMemo(() => {
    if (!result?.features) return [];
    const keys = [
      "mean_hue", "mean_saturation", "mean_value", "hue_std",
      "saturation_std", "color_diversity", "dominant_color_ratio_max",
    ];
    return keys.filter((k) => k in result.features).map((k) => ({ key: k, value: result.features[k] }));
  }, [result]);

  const gestaltFeatures = useMemo(() => {
    if (!result?.features) return [];
    return Object.entries(result.features)
      .filter(([k]) => k.endsWith("_score"))
      .map(([k, v]) => ({ key: k.replace("_score", ""), value: v }));
  }, [result]);

  return (
    <div className="min-h-screen w-full bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      <div className="mx-auto flex max-w-7xl flex-col gap-5 px-4 py-6">
        {/* Header */}
        <header className="flex flex-wrap items-center justify-between gap-4 rounded-2xl bg-slate-900/80 px-6 py-4 shadow-lg backdrop-blur">
          <div>
            <p className="text-[10px] font-medium uppercase tracking-widest text-cyan-400">
              explainable art authenticity
            </p>
            <h1 className="text-2xl font-bold text-white">Art Authenticity Inspector</h1>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <label className="flex cursor-pointer items-center gap-2 rounded-xl border border-slate-700 bg-slate-800/80 px-4 py-2 text-sm text-slate-100 transition hover:border-cyan-500">
              <input type="file" accept="image/*" className="hidden" onChange={onFileChange} />
              <svg className="h-4 w-4 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
              {file ? file.name.slice(0, 20) + (file.name.length > 20 ? "…" : "") : "画像を選択"}
            </label>
            <label className="flex cursor-pointer items-center gap-2 rounded-xl border border-slate-700 bg-slate-800/80 px-3 py-2 text-xs text-slate-200 transition hover:border-cyan-500">
              <input
                type="checkbox"
                className="h-3.5 w-3.5 accent-cyan-400"
                checked={includeGestalt}
                onChange={(e) => setIncludeGestalt(e.target.checked)}
              />
              ゲシュタルト
            </label>
            <input
              className="w-48 rounded-xl border border-slate-700 bg-slate-800/80 px-3 py-2 text-xs text-slate-100 focus:border-cyan-500 focus:outline-none"
              value={apiBase}
              onChange={(e) => setApiBase(e.target.value)}
              placeholder="API URL"
            />
            <button
              onClick={submit}
              disabled={loading}
              className="rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 px-5 py-2 text-sm font-semibold text-white shadow-lg shadow-cyan-600/30 transition hover:scale-[1.02] disabled:opacity-50"
            >
              {loading ? "推論中…" : "推論を実行"}
            </button>
          </div>
        </header>

        {error && (
          <div className="rounded-xl border border-rose-500/40 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
            {error}
          </div>
        )}

        {/* Main Grid */}
        <div className="grid gap-5 lg:grid-cols-12">
          {/* Left: Image Preview */}
          <section className="flex flex-col gap-4 rounded-2xl bg-slate-900/70 p-5 shadow-lg lg:col-span-4">
            <h2 className="text-sm font-semibold text-slate-300">画像プレビュー</h2>
            <div className="flex flex-1 items-center justify-center rounded-xl border border-dashed border-slate-700 bg-slate-950/50 p-4">
              {previewUrl ? (
                <img src={previewUrl} alt="preview" className="max-h-72 rounded-lg object-contain shadow-lg" />
              ) : (
                <p className="text-center text-sm text-slate-500">PNG / JPG をアップロード</p>
              )}
            </div>
            {/* Meta Info */}
            {result && (
              <div className="grid grid-cols-2 gap-2 text-xs text-slate-300">
                <div className="rounded-lg bg-slate-800/60 px-3 py-2">
                  <span className="text-slate-500">寸法</span>
                  <div className="font-medium text-slate-100">
                    {result.metadata?.width && result.metadata?.height
                      ? `${result.metadata.width}×${result.metadata.height}`
                      : "-"}
                  </div>
                </div>
                <div className="rounded-lg bg-slate-800/60 px-3 py-2">
                  <span className="text-slate-500">形式</span>
                  <div className="font-medium text-slate-100">{result.metadata?.format ?? "-"}</div>
                </div>
                <div className="rounded-lg bg-slate-800/60 px-3 py-2">
                  <span className="text-slate-500">サイズ</span>
                  <div className="font-medium text-slate-100">{byteFmt(result.metadata?.file_size_bytes)}</div>
                </div>
                <div className="rounded-lg bg-slate-800/60 px-3 py-2">
                  <span className="text-slate-500">ゲシュタルト</span>
                  <div className="font-medium text-slate-100">{result.gestalt_status === "ok" ? "計算済" : "未計算"}</div>
                </div>
              </div>
            )}
          </section>

          {/* Center: Result + Feature Importances */}
          <section className="flex flex-col gap-4 rounded-2xl bg-slate-900/70 p-5 shadow-lg lg:col-span-5">
            <h2 className="text-sm font-semibold text-slate-300">推論結果</h2>
            {result ? (
              <>
                <div className="flex items-center gap-4">
                  <span
                    className={`rounded-full px-4 py-1.5 text-sm font-bold ${
                      result.prediction.label === "fake"
                        ? "bg-rose-500/20 text-rose-300"
                        : "bg-emerald-500/20 text-emerald-300"
                    }`}
                  >
                    {result.prediction.label === "fake" ? "偽物 (Fake)" : "本物 (Authentic)"}
                  </span>
                  <span className="text-sm text-slate-400">
                    確信度 <span className="font-semibold text-white">{(result.prediction.confidence * 100).toFixed(1)}%</span>
                  </span>
                </div>
                <div className="flex items-center gap-3 text-xs text-slate-400">
                  <span>Authentic: {(result.prediction.probabilities.authentic * 100).toFixed(1)}%</span>
                  <span>Fake: {(result.prediction.probabilities.fake * 100).toFixed(1)}%</span>
                </div>
                <div className="h-2.5 w-full overflow-hidden rounded-full bg-slate-800">
                  <div
                    className={`h-full transition-all ${result.prediction.label === "fake" ? "bg-rose-500" : "bg-emerald-500"}`}
                    style={{ width: `${result.prediction.confidence * 100}%` }}
                  />
                </div>

                <div className="mt-2">
                  <h3 className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-400">
                    重要度の高い特徴（RF Feature Importances）
                  </h3>
                  <div className="space-y-1.5">
                    {result.prediction.feature_importances.length === 0 ? (
                      <p className="text-sm text-slate-500">情報がありません。</p>
                    ) : (
                      result.prediction.feature_importances.map((f, i) => (
                        <div key={f.feature} className="flex items-center gap-3">
                          <span className="w-5 text-right text-xs text-slate-500">{i + 1}</span>
                          <div className="flex-1">
                            <div className="h-5 overflow-hidden rounded bg-slate-800/80">
                              <div
                                className="h-full bg-gradient-to-r from-cyan-600 to-cyan-400"
                                style={{ width: `${Math.min(f.importance * 600, 100)}%` }}
                              />
                            </div>
                          </div>
                          <span className="w-44 text-xs text-slate-200">{f.feature}</span>
                          <span className="w-16 text-right text-xs text-slate-400">{numberFmt.format(f.value)}</span>
                          <span className="w-12 text-right text-xs font-semibold text-cyan-300">
                            {(f.importance * 100).toFixed(1)}%
                          </span>
                        </div>
                      ))
                    )}
                  </div>
                </div>
              </>
            ) : (
              <p className="py-8 text-center text-sm text-slate-500">推論結果がここに表示されます</p>
            )}
          </section>

          {/* Right: Color & Gestalt Features */}
          <section className="flex flex-col gap-4 rounded-2xl bg-slate-900/70 p-5 shadow-lg lg:col-span-3">
            <div>
              <h3 className="mb-2 text-sm font-semibold text-slate-300">色・テクスチャ特徴</h3>
              <div className="space-y-1">
                {colorFeatures.length === 0 ? (
                  <p className="text-xs text-slate-500">推論後に表示</p>
                ) : (
                  colorFeatures.map((c) => (
                    <div key={c.key} className="flex items-center justify-between rounded bg-slate-800/50 px-2 py-1.5">
                      <span className="text-xs text-slate-400">{c.key}</span>
                      <span className="text-xs font-medium text-slate-100">{numberFmt.format(c.value)}</span>
                    </div>
                  ))
                )}
              </div>
            </div>
            <div>
              <h3 className="mb-2 text-sm font-semibold text-slate-300">ゲシュタルト指標</h3>
              <div className="grid grid-cols-2 gap-1">
                {gestaltFeatures.length === 0 ? (
                  <p className="col-span-2 text-xs text-slate-500">推論後に表示</p>
                ) : (
                  gestaltFeatures.map((g) => (
                    <div key={g.key} className="flex items-center justify-between rounded bg-slate-800/50 px-2 py-1.5">
                      <span className="text-xs text-slate-400">{g.key}</span>
                      <span className="text-xs font-semibold text-slate-100">{Math.round(g.value)}</span>
                    </div>
                  ))
                )}
              </div>
            </div>
            {result && (
              <div className="mt-auto rounded-lg bg-slate-800/40 px-3 py-2 text-[10px] text-slate-500">
                Model: {result.model_version}
              </div>
            )}
          </section>
        </div>

        {/* History */}
        <section className="rounded-2xl bg-slate-900/70 p-5 shadow-lg">
          <div className="mb-3 flex items-center justify-between">
            <h3 className="text-sm font-semibold text-slate-300">推論履歴</h3>
            <span className="text-xs text-slate-500">{history.length} 件</span>
          </div>
          {history.length === 0 ? (
            <p className="text-sm text-slate-500">まだ履歴がありません</p>
          ) : (
            <div className="flex gap-3 overflow-x-auto pb-2">
              {history.map((item) => (
                <div
                  key={item.id}
                  className="flex w-36 flex-none flex-col rounded-xl border border-slate-800 bg-slate-950/60 p-2.5"
                >
                  {item.preview && (
                    <img src={item.preview} alt="" className="mb-2 h-20 w-full rounded-lg object-cover" />
                  )}
                  <span
                    className={`self-start rounded-full px-2 py-0.5 text-[10px] font-semibold ${
                      item.label === "fake" ? "bg-rose-500/20 text-rose-300" : "bg-emerald-500/20 text-emerald-300"
                    }`}
                  >
                    {item.label === "fake" ? "Fake" : "Auth"}
                  </span>
                  <span className="mt-1 text-xs text-slate-300">{(item.confidence * 100).toFixed(1)}%</span>
                  <span className="text-[10px] text-slate-500">{item.timestamp}</span>
                </div>
              ))}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}

export default App;
