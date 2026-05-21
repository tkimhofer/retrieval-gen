"use client";

import React, { useEffect, useMemo, useState, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import dynamic from "next/dynamic";
const CytoscapeComponent = dynamic(() => import("react-cytoscapejs"), { ssr: false });
import { Calendar, Loader2, Network, Search, Filter, ChevronsUpDown, Check, ExternalLink } from "lucide-react";
import { Popover, PopoverTrigger, PopoverContent } from "@/components/ui/popover";
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList, CommandSeparator } from "@/components/ui/command";
import { cn } from "@/lib/utils";
// import ChainlitWidget from "@/components/ui/chainlitWidget";


/**
 * Ratsinfo Public Explorer
 * - Fetches aggregated data about Referenzvorlagen and TOPs for a selected Gremium and time range
 * - Shows: summary cards, table, and an interactive graph (Referenzvorlage ↔ TOP)
 * - Ready to drop into a Next.js app (App Router, src/ directory)
 */

const API_BASE = process.env.NEXT_PUBLIC_RATSINFO_API || ""; // set in .env.local
const TOP_LABEL_MAX = 40; // max characters for TOP titles in graph labels


// --- Types (JSDoc for editor help) ---
/** @typedef {{ top_id: string, name?: string, meeting?: string, vorlage_url?: string, vorlage?: string, meeting_date?: string }} TopBrief */
/** @typedef {{ referenzvorlage: string, num_tops: number, num_meetings?: number, tops?: TopBrief[] }} RefRow */

const demoData = /** @type {RefRow[]} */ ([
  { referenzvorlage: "BV-2024-001", num_tops: 3, num_meetings: 2, tops: [
    { top_id: "T-1001", name: "Kita-Ausbau", meeting: "Rat 2024-05-15", meeting_date: "2024-05-15" },
    { top_id: "T-1022", name: "Kita-Ausbau (Fortsetzung)", meeting: "Rat 2024-06-20", meeting_date: "2024-06-20" },
    { top_id: "T-1099", name: "Kita-Mittel", meeting: "Rat 2024-09-18", meeting_date: "2024-09-18" },
  ]},
  { referenzvorlage: "BV-2024-017", num_tops: 2, num_meetings: 1, tops: [
    { top_id: "T-1201", name: "Radwege Innenstadt", meeting: "Rat 2024-10-12", meeting_date: "2024-10-12" },
    { top_id: "T-1210", name: "Radwege Innenstadt (Änderungsantrag)", meeting: "Rat 2024-10-12", vorlage: "BV-2024-017", vorlage_url: "https://sessionnet.owl-it.de/duisburg/bi/vo0050.asp?__kvonr=123456" },
  ]},
]);

// Grouped, searchable Gremium list (Combobox)
const GREMIA_GROUPS: { label: string; items: string[] }[] = [
  {
    label: "Hauptorgane & Aufsichtsgremien",
    items: [
      "Rat der Stadt",
      "Haupt- und Finanzausschuss",
      "Rechnungsprüfungsausschuss",
    ],
  },

   {
    label: "Verwaltung & Soziales",
    items: [
      "Ausschuss für Ordnungs- und Bürgerangelegenheiten",
      "Ausschuss für Arbeit, Soziales und Gesundheit",
      "Gleichstellungsausschuss",
      "Integrationsrat",
      "Beirat für Menschen mit Behinderungen",
      "Seniorenbeirat",
    ],
  },
 {
    label: "Digitalisierung, Stadtentwicklung & Umwelt",
    items: [
      "Digitalisierungsausschuss",
      "Ausschuss für Stadtentwicklung und Verkehr",
      "Ausschuss für Umwelt, Klima und Naturschutz",
    ],
  },
  {
    label: "Kultur & Jugend",
    items: [
      "Kulturausschuss",
      "Jugendhilfeausschuss",
    ],
  },
  {
    label: "Steuerung & Organisation",
    items: [
      "Vergabeausschuss",
      "Wahlausschuss",
      "Wahlprüfungsausschuss"
    ],
  },
 {
    label: "Betriebsausschüsse",
    items: [
      "Betriebsausschuss DuisburgSport",
      "Betriebsausschuss für das Immobilien-Management Duisburg",
      "Betriebsausschuss für städtische Immobilien"
    ],
  },

//   {
//     label: "Ausschüsse",
//     items: [
//       "Jugendhilfeausschuss",
//       "Vergabeausschuss",
//       "Betriebsausschuss DuisburgSport",
//       "Betriebsausschuss für städtische Immobilien",
//       "Ausschuss für Umwelt, Klima und Naturschutz",
//       "Ausschuss für Personal und Verwaltung",
//       "Ausschuss für Wirtschaft, Innovation und Tourismus",
//       "Schulausschuss",
//       "Digitalisierungsausschuss",
//       "Unterausschuss Universität des Schulausschusses",
//       "Rechnungsprüfungsausschus",
//       "Gleichstellungsausschuss",
//     ],
//   },
  {
    label: "Bezirksvertretungen",
    items: [
      "Bezirksvertretung Mitte",
      "Bezirksvertretung Rheinhausen",
      "Bezirksvertretung Meiderich/Beeck",
      "Bezirksvertretung Süd",
      "Bezirksvertretung Hamborn",
      "Bezirksvertretung Homberg/Ruhrort/Baerl",
      "Bezirksvertretung Walsum",
    ],
  },
//   {
//     label: "Beiräte/Räte",
//     items: [
//       "Integrationsrat",
//       "Beirat der Unteren Naturschutzbehörde der Stadt Duisburg",
//       "Beirat für Menschen mit Behinderungen",
//       "Seniorenbeirat",
//     ],
//   },
];

function useFetchReferenzvorlagen({ gremium, minTops, from, to, useDemo } : { gremium:string, minTops:number, from:string, to:string, useDemo:boolean }) {
  const [data, setData] = useState<RefRow[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    async function run() {
      setLoading(true); setError(null);
      try {
        if (useDemo || !API_BASE) {
          await new Promise(r => setTimeout(r, 200));
          if (!alive) return;
          setData(demoData.filter(d => d.num_tops >= minTops));
        } else {
          const params = new URLSearchParams();
          if (gremium) params.set("gremium", gremium);
          params.set("min_tops", String(minTops));
          if (from) params.set("from", from);
          if (to) params.set("to", to);
          const res = await fetch(`${API_BASE}/api/referenzvorlagen?${params.toString()}`);
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          const json = await res.json();
          if (!alive) return;
          setData(json);
        }
      } catch (e:any) {
        if (!alive) return;
        setError(e?.message ?? String(e));
      } finally {
        if (alive) setLoading(false);
      }
    }
    run();
    return () => { alive = false; };
  }, [gremium, minTops, from, to, useDemo]);

  return { data, loading, error };
}

function buildGraphElements(rows: RefRow[]) {
  const elements: any[] = [];
  const seen = new Set<string>();
  for (const r of rows) {
    const refId = `r:${r.referenzvorlage}`;
    if (!seen.has(refId)) {
      elements.push({ data: { id: refId, label: r.referenzvorlage, type: "referenz" } });
      seen.add(refId);
    }
    if (Array.isArray(r.tops)) {
      for (const t of r.tops) {
        const tid = `t:${t.top_id}`;
        if (!seen.has(tid)) {
          const d = t.meeting_date ? String(t.meeting_date).slice(0, 10) : null;
          // Prefer date + vorlage; fall back to vorlage; finally to top_id
          const label = d && t.vorlage
            ? `${d} · ${t.vorlage}`
            : (d || t.vorlage || t.top_id || "");
          elements.push({ data: { id: tid, label, type: "top" } });
          seen.add(tid);
        }
        elements.push({ data: { id: `${tid}::${r.referenzvorlage}`, source: tid, target: refId } });
      }
    }
  }
  return elements;
}

function GremiumCombobox({ value, onChange }:{ value:string; onChange:(v:string)=>void }) {
  const [open, setOpen] = useState(false);
  const flat = useMemo(() => GREMIA_GROUPS.flatMap(g => g.items), []);

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button variant="outline" role="combobox" aria-expanded={open} className="w-full justify-between">
          {value || "Gremium wählen"}
          <ChevronsUpDown className="ml-2 h-4 w-4 opacity-50" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="p-0 w-[--radix-popover-trigger-width]">
        <Command filter={(val, search) => (val.toLowerCase().includes(search.toLowerCase()) ? 1 : 0)}>
          <CommandInput placeholder="Gremium suchen…" />
          <CommandEmpty>Kein Treffer</CommandEmpty>
          <CommandList>
            {GREMIA_GROUPS.map((group, gi) => (
              <React.Fragment key={group.label}>
                <CommandGroup heading={group.label}>
                  {group.items.map((item) => (
                    <CommandItem
                      key={item}
                      value={item}
                      onSelect={() => { onChange(item); setOpen(false); }}
                    >
                      <Check className={cn("mr-2 h-4 w-4", value === item ? "opacity-100" : "opacity-0")} />
                      {item}
                    </CommandItem>
                  ))}
                </CommandGroup>
                {gi < GREMIA_GROUPS.length - 1 ? <CommandSeparator /> : null}
              </React.Fragment>
            ))}
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  );
}

export default function RatsinfoExplorer() {
  const [gremium, setGremium] = useState("Rat der Stadt");
  const [minTops, setMinTops] = useState(2);
  const [from, setFrom] = useState(""); // YYYY-MM-DD
  const [to, setTo] = useState("");
  const [useDemo, setUseDemo] = useState(true);
  const [tab, setTab] = useState("refs");
  const [mainTab, setMainTab] = useState("search");

  const { data, loading, error } = useFetchReferenzvorlagen({ gremium, minTops, from, to, useDemo });
  const graphElements = useMemo(() => (data ? buildGraphElements(data) : []), [data]);

  // Respect backend ordering: no client-side resorting
  const rows = data ?? [] as RefRow[];

  // --- Full-text Search (Phase 1) ---
  const [searchQ, setSearchQ] = useState("");
  const [searchScope] = useState<"top">("top");
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [searchData, setSearchData] = useState<SearchResponse | null>(null);


  type Result = {
      qtype?: string;
      chunk_id?: string;
      vorlage_id?: string;
      aktenzeichen?: string;
      typ?: string | null;
      betreff?: string;
      text?: string;
      text_betr?: string;
      hit_score_norm?: number;
      hit_score?: number;
      hit_id?: string | null;
      source?: string;
      datum_min?: string;
      datum_max?: string;
  };


  type SearchItem = [number, Result]; // [rerankScore, result]
  type SearchResponse = { query: string; items: SearchItem[]; took_ms: number };

  const items = (searchData?.items ?? []) as SearchItem[];

  console.log("searchData", searchData);
  console.log("first item", searchData?.items?.[0]);



  const API_BASE = process.env.NEXT_PUBLIC_RATSINFO_API; // e.g. "http://192.168.178.84:8081"
  const controllerRef = useRef<AbortController | null>(null);

  async function doSearch() {

    setSearchError(null);
    if (!API_BASE) { setSearchError("API nicht konfiguriert (NEXT_PUBLIC_RATSINFO_API fehlt)."); return; }
    if (!searchQ.trim()) { setSearchError("Bitte Suchbegriff eingeben."); return; }
    setSearchLoading(true);

    // cancel any in-flight request
    controllerRef.current?.abort();
    const ac = new AbortController();
    controllerRef.current = ac;

      // If you use a Next.js proxy/rewrites, set:
      // const url = "/api/search";
//       const url = `${API_BASE.replace(/\/+$/, "")}/search`;
//       const url = "http://192.168.178.84:8081/search"

      try {
        const res = await fetch("http://192.168.10.157:8081/search", {
          method: "POST",
          signal: ac.signal,
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query: searchQ.trim(),
            retrieve_k: 50,
            top_k: 10,
            include_text: true,
          }),
          // mode: "cors", // optional; browsers default to CORS for cross-origin
        });

        const isJson = res.headers.get("content-type")?.includes("application/json");

        if (!res.ok) {
          let msg = `${res.status} ${res.statusText}`;
          if (isJson) {
            try {
              const err = await res.json();
              const detail = err?.detail ?? err?.message;
              if (detail) msg += ` — ${detail}`;
            } catch {}
          } else {
            try {
              const t = await res.text();
              if (t) msg += ` — ${t}`;
            } catch {}
          }
          throw new Error(msg);
        }

        const json = (await res.json()) as SearchResponse;
        setSearchData(json);
      } catch (e: any) {
        if (e?.name !== "AbortError") setSearchError(e?.message ?? String(e));
      } finally {
        if (controllerRef.current === ac) controllerRef.current = null;
        setSearchLoading(false);
      }
  }


//   async function doSearch() {
//     setSearchError(null);
//     if (!API_BASE) { setSearchError("API nicht konfiguriert (NEXT_PUBLIC_RATSINFO_API fehlt)."); return; }
//     if (!searchQ.trim()) { setSearchError("Bitte Suchbegriff eingeben."); return; }
//     setSearchLoading(true);
//
//     try {
//       const res = await fetch("http://192.168.178.84:8081/search", {
//         method: "POST",
// //         signal: controllerRef.current.signal,
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify({
//           query: searchQ.trim(),
//           retrieve_k: 50,
//           top_k: 10,
//           include_text: true
//         }),
//       });
//       if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
//       const json = (await res.json()) as SearchResponse;
//       setSearchData(json);
//     } catch (e: any) {
//       setSearchError(e?.message ?? String(e));
//     } finally {
//       setSearchLoading(false);
//     }
//   }

  async function runSearch() {
    setSearchError(null);
    if (!API_BASE) { setSearchError("API nicht konfiguriert (NEXT_PUBLIC_RATSINFO_API fehlt)."); return; }
    if (!searchQ.trim()) { setSearchError("Bitte Suchbegriff eingeben."); return; }
    setSearchLoading(true);
    try {
      const params = new URLSearchParams({ q: searchQ, scope: "top" });
      if (gremium) params.set("gremium", gremium);
//       const res = await fetch(`${API_BASE}/search?${params.toString()}`);
      const res = await fetch(`${API_BASE}/api/search?${params.toString()}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      setSearchData(json);
    } catch (e: any) {
      setSearchError(e?.message ?? String(e));
    } finally {
      setSearchLoading(false);
    }
  }

  // Build share URL on the client only to avoid hydration mismatch
  const [shareUrl, setShareUrl] = useState("");
  useEffect(() => {
    if (typeof window === 'undefined') return;
    const u = new URL(window.location.href);
    u.searchParams.set("gremium", gremium);
    u.searchParams.set("minTops", String(minTops));
    if (from) u.searchParams.set("from", from); else u.searchParams.delete("from");
    if (to) u.searchParams.set("to", to); else u.searchParams.delete("to");
    setShareUrl(u.toString());
  }, [gremium, minTops, from, to]);

  return (
    <div className="max-w-6xl mx-auto p-4 md:p-8 space-y-6">
      <div className="flex items-center justify-between gap-4">
        <h1 className="text-2xl md:text-3xl font-semibold tracking-tight">Duisburger Ratsinformation</h1>
         <p className="text-muted-foreground mt-1">
    Suche, Analyse und Vernetzung kommunaler Gremiendokumente</p>
      </div>

      <Tabs value={mainTab} onValueChange={setMainTab} defaultValue="search" className="mt-2">
        <TabsList>
          <TabsTrigger value="search"><Search className="h-4 w-4 mr-1"/>Suche</TabsTrigger>
          <TabsTrigger value="explorer"><Network className="h-4 w-4 mr-1"/>Explorer</TabsTrigger>
        </TabsList>

        <TabsContent value="search" className="mt-4">
          <Card className="rounded-2xl">
            <CardHeader className="pb-2"><CardTitle>Volltext-Suche (TOPs)

            </CardTitle></CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center gap-2 flex-wrap">
                <div className="w-64 md:w-72">
                  <GremiumCombobox value={gremium} onChange={setGremium} />
                </div>
                <Input
                  suppressHydrationWarning
                  placeholder={`Suchbegriff (z.B. DS-2026-001*, "Bundesgarteschau", radweg~1)`}
                  value={searchQ}
                  onChange={(e) => setSearchQ(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter') doSearch(); }}
                  className="w-64 md:w-80"
                />
                <Button onClick={doSearch} disabled={searchLoading}>
                  {searchLoading ? <Loader2 className="h-4 w-4 mr-2 animate-spin"/> : <Search className="h-4 w-4 mr-2"/>}
                  Suchen
                </Button>
              </div>
              {searchError && <div className="text-destructive text-sm">Fehler: {searchError}</div>}

              <div>
                <h3 className="text-sm font-medium mb-2">TOPs</h3>
                {searchLoading ?(
                  <div className="text-sm text-muted-foreground">Suche läuft…</div>
                ) : !searchData ? (
                      <div className="text-sm text-muted-foreground">Noch keine Suche.</div>
                    ) : ( // we have a response(
                <ul className="space-y-2">
                    {items.length === 0 ? (
                    <li className="p-2 text-sm text-muted-foreground">Keine Ergebnisse.</li>
                  ) : (
                    items.map(([score, r], i) => {
                      const key = r.chunk_id ?? r.hit_id ?? `${r.vorlage_id ?? "row"}-${i}`;
//                       const title =
//                         r.betreff || r.vorlage_id || r.aktenzeichen || r.chunk_id || "Result";
                      const title = r.betreff ;
                      const snippet = r.text ?? "";
                      const dateRange =
                        r.datum_min && r.datum_max
                          ? `${r.datum_min} – ${r.datum_max}`
                          : r.datum_min || r.datum_max || null;

                      return (
                        <li key={key} className="p-2 rounded border">
                        <div className="font-medium">Vorlage {r.vorlage_id}</div>
                        <div className="font-medium">{title}</div>

                          {snippet && (
                            <p className="mt-1 text-sm whitespace-pre-wrap">{snippet}</p>
                          )}

                          <div className="mt-2 text-xs text-muted-foreground flex flex-wrap gap-2">
                            {r.qtype && <span className="px-2 py-0.5 rounded border" color="red">{r.qtype}</span>}
                            <span className="px-2 py-0.5 rounded border">
                              score {Number.isFinite(score) ? score.toFixed(3) : String(score)}
                            </span>
                            {r.aktenzeichen && (
                              <span className="px-2 py-0.5 rounded border">
                                {r.aktenzeichen}
                              </span>
                            )}
                            {dateRange && (
                              <span className="px-2 py-0.5 rounded border">{dateRange}</span>
                            )}
                          </div>
                        </li>
                      );
                    })
                  )}
                </ul>
//                   <ul className="space-y-2">
//                     {searchData.items.map(([rtype, rank, score, text]: [string, number, number, string], i: number) => (
//                       <li key={`${rtype}-${rank}-${i}`} className="p-2 rounded border">
//                       <div className="text-sm whitespace-pre-wrap">{text}</div>
//
//                       <div className="mt-2 text-xs text-muted-foreground flex flex-wrap gap-2">
//                         <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded border">
//                           {rtype}
//                         </span>
//                         <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded border">
//                           rank {rank}
//                         </span>
//                         <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded border">
//                           {Number.isFinite(score) ? score.toFixed(3) : String(score)}
//                         </span>
//                       </div>
//                     </li>
//                     ))}
//                   </ul>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="explorer" className="mt-4">
          <Card className="rounded-2xl mb-4">
            <CardHeader className="pb-2"><CardTitle>Filter</CardTitle></CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
                <div>
                  <Label>Gremium</Label>
                  <GremiumCombobox value={gremium} onChange={setGremium} />
                </div>
                <div>
                  <Label>Von</Label>
                  <Input placeholder="2026-01-01" value={from} onChange={(e)=>setFrom(e.target.value)} />
                </div>
                <div>
                  <Label>Bis</Label>
                  <Input placeholder="2026-03-31" value={to} onChange={(e)=>setTo(e.target.value)} />
                </div>
                <div>
                  <Label>min. TOPs</Label>
                  <Input placeholder="2" value={String(minTops)} onChange={(e)=>setMinTops(parseInt(e.target.value || '0') || 0)} />
                </div>
              </div>
              <div className="mt-2 flex items-center gap-2">
                <Button variant={useDemo ? "default" : "outline"} onClick={() => setUseDemo(!useDemo)}>
                  {useDemo ? "Demo-Daten" : "Live-API"}
                </Button>
                {shareUrl && <a href={shareUrl} className="text-sm text-muted-foreground underline" target="_blank" rel="noreferrer">Teilen</a>}
              </div>
            </CardContent>
          </Card>

          <Tabs value={tab} onValueChange={setTab} defaultValue="refs">
  <TabsList>
    <TabsTrigger value="refs">Referenzvorlagen</TabsTrigger>
    <TabsTrigger value="graph"><Network className="h-4 w-4 mr-1"/>Graph</TabsTrigger>
  </TabsList>


        <TabsContent value="refs" className="mt-4">
          <Card className="rounded-2xl">
            <CardContent className="p-0 overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-muted/30">
                  <tr>
                    <th className="text-left p-3">Referenzvorlage</th>
                    <th className="text-left p-3"># TOPs</th>
                    <th className="text-left p-3">Meetings</th>
                    <th className="text-left p-3">Details</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row) => (
                    <tr key={row.referenzvorlage} className="border-t hover:bg-muted/20">
                      <td className="p-3 font-medium">{row.referenzvorlage}</td>
                      <td className="p-3">{row.num_tops}</td>
                      <td className="p-3">{row.num_meetings ?? "—"}</td>
                      <td className="p-3">
                        {Array.isArray(row.tops) && row.tops.length > 0 ? (
                          <details>
                            <summary className="cursor-pointer text-primary">TOPs anzeigen</summary>
                            <ul className="pl-5 list-disc mt-2 space-y-1">
                              {row.tops.map(t => (
                                <li key={t.top_id} className="text-muted-foreground flex flex-col gap-0.5">
                                  <div className="font-medium text-foreground">{t.name || t.top_id}</div>
                                  {t.meeting_date ? (
                                    <div className="text-muted-foreground">{String(t.meeting_date).slice(0, 10)}</div>
                                  ) : null}
                                  {t.vorlage ? (
                                    t.vorlage_url ? (
                                      <a
                                        href={t.vorlage_url}
                                        className="inline-flex items-center gap-1 underline"
                                        target="_blank"
                                        rel="noreferrer"
                                      >
                                        Vorlage {t.vorlage} <ExternalLink className="h-3 w-3" />
                                      </a>
                                    ) : (
                                      <div className="text-muted-foreground">Vorlage {t.vorlage}</div>
                                    )
                                  ) : null}
                                </li>
                              ))}
                            </ul>
                          </details>
                        ) : (
                          <span className="text-muted-foreground">—</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="graph" className="mt-4">
          <Card className="rounded-2xl">
            <CardHeader className="pb-2"><CardTitle>Referenzvorlage ↔ TOP</CardTitle></CardHeader>
            <CardContent>
              {graphElements.length === 0 ? (
                <div className="text-sm text-muted-foreground">Keine Kanten darstellbar. API muss TOP-Listen je Referenzvorlage liefern (oder Demo-Daten aktivieren).</div>
              ) : (
                <div className="h-[520px]">
                  <CytoscapeComponent
                    elements={graphElements}
                    layout={{ name: "cose", animate: false }}
                    style={{ width: "100%", height: "100%" }}
                    stylesheet={[
                      { selector: 'node', style: { 'label': 'data(label)', 'font-size': 12, 'text-valign': 'center', 'text-halign': 'center', 'background-opacity': 0.9 } },
                      { selector: 'node[type="referenz"]', style: { 'shape': 'round-rectangle', 'padding': '8px', 'font-weight': 600 } },
                      { selector: 'node[type="top"]', style: { 'shape': 'ellipse' } },
                      { selector: 'edge', style: { 'curve-style': 'bezier', 'target-arrow-shape': 'triangle', 'width': 2 } },
                    ]}
                  />
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
        </TabsContent>
      </Tabs>

      {/* Footer */}
      <div className="text-xs text-muted-foreground">
        <p>
          API erwartet: GET <code>/referenzvorlagen?gremium=&lt;str&gt;&amp;min_tops=&lt;int&gt;&amp;from=&lt;YYYY-MM-DD&gt;&amp;to=&lt;YYYY-MM-DD&gt;</code>
        </p>
        <p>Optional: Endpoint liefert <code>tops[]</code> je Referenzvorlage, um den Graphen zu zeichnen.</p>
      </div>
    </div>
  );
}
