from __future__ import annotations

from datetime import datetime
from html import escape
from typing import Any

import ipywidgets as widgets
import pandas as pd
from IPython.display import HTML, Markdown, clear_output, display

from .config import PipelineConfig
from .pipeline import persist_results, run_full_analysis, run_ingest_and_index


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _confidence_badge(value: Any) -> str:
    score = _coerce_float(value, 0.5)
    if score >= 0.75:
        color, label = "#0f766e", "High"
    elif score >= 0.55:
        color, label = "#b45309", "Medium"
    else:
        color, label = "#b91c1c", "Low"
    return f"<span style='padding:2px 8px;border-radius:12px;background:{color};color:white;font-size:12px;'>{label} ({score:.2f})</span>"


def _topic_label(topic: str) -> str:
    mapping = {
        "policy_rates": "Policy Rates",
        "financial_conditions": "Financial Conditions",
    }
    return mapping.get(topic, str(topic).replace("_", " ").title())


def launch_investor_dashboard(cfg: PipelineConfig):
    state: dict[str, Any] = {
        "cfg": cfg,
        "ingest_index_result": None,
        "analysis_result": None,
        "last_saved": None,
    }

    profile_dd = widgets.Dropdown(
        options=[
            ("Fast default (180d, 24 PDFs)", "fast_default"),
            ("Full default (540d, 40 PDFs)", "full_default"),
        ],
        value=cfg.profile_name,
        description="Profile:",
        layout=widgets.Layout(width="360px"),
    )
    run_mode_dd = widgets.Dropdown(
        options=[
            ("Refresh ingestion + index + analysis", "refresh_all"),
            ("Analysis only (use existing index)", "analysis_only"),
        ],
        value="analysis_only",
        description="Run mode:",
        layout=widgets.Layout(width="420px"),
    )
    topic_focus_dd = widgets.Dropdown(
        options=[
            ("All topics", "all"),
            ("Inflation", "inflation"),
            ("Unemployment", "unemployment"),
            ("Growth", "growth"),
            ("Policy rates", "policy_rates"),
            ("Financial conditions", "financial_conditions"),
            ("Credit", "credit"),
        ],
        value="all",
        description="Focus:",
        layout=widgets.Layout(width="320px"),
    )

    top_k_topic_slider = widgets.IntSlider(value=2, min=1, max=4, step=1, description="Top-k/topic:", continuous_update=False)
    max_context_slider = widgets.IntSlider(
        value=2400, min=1200, max=4200, step=200, description="Context chars:", continuous_update=False
    )
    num_predict_slider = widgets.IntSlider(value=380, min=180, max=700, step=20, description="LLM tokens:", continuous_update=False)
    reranker_toggle = widgets.Checkbox(value=False, description="Enable reranker")

    run_btn = widgets.Button(description="Run", button_style="primary", icon="play")
    save_btn = widgets.Button(description="Save outputs", icon="save")

    log_out = widgets.Output(layout=widgets.Layout(border="1px solid #ddd", padding="8px"))
    dash_out = widgets.Output(layout=widgets.Layout(border="1px solid #ddd", padding="8px"))
    topic_out = widgets.Output(layout=widgets.Layout(border="1px solid #ddd", padding="8px"))
    json_out = widgets.Output(layout=widgets.Layout(border="1px solid #ddd", padding="8px"))

    tabs = widgets.Tab(children=[dash_out, topic_out, json_out])
    tabs.set_title(0, "Dashboard")
    tabs.set_title(1, "Topic Drilldown")
    tabs.set_title(2, "Normalized JSON")

    def apply_controls_to_cfg() -> PipelineConfig:
        run_cfg = state["cfg"]
        run_cfg.set_profile(profile_dd.value)
        run_cfg.top_k_topic = int(top_k_topic_slider.value)
        run_cfg.max_context_chars = int(max_context_slider.value)
        run_cfg.ollama_num_predict = int(num_predict_slider.value)
        run_cfg.enable_reranker = bool(reranker_toggle.value)
        run_cfg.enforce_topic_min_evidence = True
        return run_cfg

    def render_dashboard() -> None:
        analysis = state.get("analysis_result")
        ingest = state.get("ingest_index_result")

        with dash_out:
            clear_output()
            if not analysis:
                display(Markdown("Run the pipeline to populate the investor dashboard."))
                return

            parsed = analysis.get("parsed") or {}
            regime = parsed.get("regime_call", {}) if isinstance(parsed, dict) else {}
            timings = analysis.get("timings", {})
            counts = analysis.get("analysis_counts", {})
            stage_metrics = (ingest or {}).get("stage_metrics", {}) if ingest else {}

            headline = escape(str(parsed.get("executive_summary", "No summary generated yet.")))
            growth = escape(str(regime.get("growth_momentum", "n/a")))
            inflation = escape(str(regime.get("inflation_trend", "n/a")))
            policy = escape(str(regime.get("policy_bias", "n/a")))
            recession = escape(str(regime.get("recession_risk", "n/a")))
            conf_badge = _confidence_badge(regime.get("confidence", 0.5))

            cards = (
                "<div style='font-family:Arial,sans-serif;'>"
                "<div style='padding:12px 14px;border:1px solid #ddd;border-radius:10px;margin-bottom:10px;background:#fafafa;'>"
                "<div style='font-size:13px;color:#555;margin-bottom:6px;'>Investor Brief (6-12m)</div>"
                f"<div style='font-size:16px;line-height:1.45;'>{headline}</div>"
                "</div>"
                "<div style='display:grid;grid-template-columns:repeat(5,minmax(140px,1fr));gap:8px;margin-bottom:10px;'>"
                f"<div style='border:1px solid #ddd;border-radius:8px;padding:8px;'><div style='font-size:12px;color:#666;'>Growth</div><div style='font-size:15px;'>{growth}</div></div>"
                f"<div style='border:1px solid #ddd;border-radius:8px;padding:8px;'><div style='font-size:12px;color:#666;'>Inflation</div><div style='font-size:15px;'>{inflation}</div></div>"
                f"<div style='border:1px solid #ddd;border-radius:8px;padding:8px;'><div style='font-size:12px;color:#666;'>Policy bias</div><div style='font-size:15px;'>{policy}</div></div>"
                f"<div style='border:1px solid #ddd;border-radius:8px;padding:8px;'><div style='font-size:12px;color:#666;'>Recession risk</div><div style='font-size:15px;'>{recession}</div></div>"
                f"<div style='border:1px solid #ddd;border-radius:8px;padding:8px;'><div style='font-size:12px;color:#666;'>Confidence</div><div style='font-size:15px;'>{conf_badge}</div></div>"
                "</div></div>"
            )
            display(HTML(cards))

            if stage_metrics:
                display(Markdown("### Run Metrics"))
                metrics = {
                    "profile": stage_metrics.get("profile_name", "n/a"),
                    "catalog_candidates": stage_metrics.get("catalog_candidates", "n/a"),
                    "downloaded_or_exists": stage_metrics.get("downloaded_or_exists", "n/a"),
                    "chunks": stage_metrics.get("chunks", "n/a"),
                    "ingest_latency_s": round(stage_metrics.get("ingest_latency_s", 0), 2),
                    "index_latency_s": round(stage_metrics.get("index_latency_s", 0), 2),
                }
                display(pd.DataFrame([metrics]))

            llm_metrics = {
                "retrieved_unique_chunks": counts.get("retrieved_unique_chunks", "n/a"),
                "llm_stage_s": round(timings.get("llm_stage_s", 0), 2),
                "retrieval_latency_s": round(timings.get("retrieval_latency_s", 0), 2),
            }
            display(Markdown("### Analysis Metrics"))
            display(pd.DataFrame([llm_metrics]))

            if analysis.get("analysis_warnings"):
                display(Markdown("### Warnings"))
                for warning in analysis["analysis_warnings"]:
                    display(Markdown(f"- {warning}"))

    def render_topic_drilldown() -> None:
        analysis = state.get("analysis_result")

        with topic_out:
            clear_output()
            if not analysis or not analysis.get("parsed"):
                display(Markdown("Run analysis to view topic drilldown."))
                return

            parsed = analysis["parsed"]
            selected = topic_focus_dd.value
            topic_signals = parsed.get("topic_signals", [])
            citations = analysis.get("citation_preview_df", pd.DataFrame())

            rows = []
            topic_to_ids: dict[str, list[str]] = {}
            for signal in topic_signals:
                topic = signal.get("topic", "")
                if selected != "all" and topic != selected:
                    continue
                evidence = signal.get("evidence", []) if isinstance(signal.get("evidence", []), list) else []
                topic_to_ids[topic] = evidence
                rows.append(
                    {
                        "topic": _topic_label(topic),
                        "view": signal.get("view", ""),
                        "confidence": signal.get("confidence", ""),
                        "evidence_count": len(evidence),
                    }
                )

            display(Markdown("### Topic Signals"))
            if rows:
                display(pd.DataFrame(rows))
            else:
                display(Markdown("No topic rows available for the selected focus."))
                return

            display(Markdown("### Evidence Quotes"))
            if citations is None or citations.empty:
                display(Markdown("No citation preview available."))
                return

            if selected == "all":
                display(citations[["doc_id", "chunk_id", "quote"]].head(12))
            else:
                ids = set(topic_to_ids.get(selected, []))
                subset = citations[citations["chunk_id"].isin(ids)]
                if subset.empty:
                    display(Markdown("No quote rows matched this topic selection."))
                else:
                    display(subset[["doc_id", "chunk_id", "quote"]].head(12))

            takeaways = parsed.get("investor_takeaways", [])
            if takeaways:
                display(Markdown("### Investor Takeaways"))
                for item in takeaways:
                    horizon = escape(str(item.get("horizon", "6-12 months")))
                    thesis = escape(str(item.get("thesis", "")))
                    display(HTML(f"<div style='margin-bottom:8px;'><b>{horizon}</b>: {thesis}</div>"))

    def render_json() -> None:
        analysis = state.get("analysis_result")
        with json_out:
            clear_output()
            if not analysis:
                display(Markdown("Run analysis to view normalized JSON output."))
                return
            print(analysis.get("normalized_json_text", analysis.get("llm_text", "")))

    def render_all() -> None:
        render_dashboard()
        render_topic_drilldown()
        render_json()

    def on_run_clicked(_):
        run_cfg = apply_controls_to_cfg()
        run_btn.disabled = True
        save_btn.disabled = True

        with log_out:
            clear_output()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting run with profile={run_cfg.profile_name}, mode={run_mode_dd.value}")
            try:
                if run_mode_dd.value == "refresh_all":
                    print("Running ingestion + indexing...")
                    state["ingest_index_result"] = run_ingest_and_index(run_cfg)
                    stage_metrics = state["ingest_index_result"].get("stage_metrics", {})
                    print("Ingestion/index complete:", stage_metrics)
                    for warning in stage_metrics.get("warnings", []):
                        print("[warn]", warning)
                else:
                    print("Skipping ingestion/index; using existing local index.")

                print("Running analysis...")
                state["analysis_result"] = run_full_analysis(run_cfg)
                print("Analysis completed. LLM latency:", round(state["analysis_result"]["timings"].get("llm_stage_s", 0), 2), "s")
                for warning in state["analysis_result"].get("analysis_warnings", []):
                    print("[warn]", warning)
            except Exception as e:
                print("Run failed:", e)
            finally:
                render_all()
                run_btn.disabled = False
                save_btn.disabled = False

    def on_save_clicked(_):
        analysis = state.get("analysis_result")
        if not analysis:
            with log_out:
                print("No analysis result to save yet. Run first.")
            return

        saved = persist_results(state["cfg"], analysis)
        state["last_saved"] = saved
        with log_out:
            print("Saved:", saved)

    def on_topic_changed(_):
        render_topic_drilldown()

    run_btn.on_click(on_run_clicked)
    save_btn.on_click(on_save_clicked)
    topic_focus_dd.observe(on_topic_changed, names="value")

    control_row_1 = widgets.HBox([profile_dd, run_mode_dd, topic_focus_dd])
    control_row_2 = widgets.HBox([top_k_topic_slider, max_context_slider, num_predict_slider, reranker_toggle])
    control_row_3 = widgets.HBox([run_btn, save_btn])

    ui = widgets.VBox(
        [
            widgets.HTML(
                "<h3 style='margin:0 0 8px 0;'>Investor Console</h3>"
                "<div style='color:#555;margin-bottom:8px;'>Choose scope, run the pipeline, and drill into topic-specific evidence quotes.</div>"
            ),
            control_row_1,
            control_row_2,
            control_row_3,
            widgets.HTML("<hr style='margin:10px 0;'/>"),
            widgets.HTML("<b>Run Log</b>"),
            log_out,
            widgets.HTML("<b>Results</b>"),
            tabs,
        ]
    )

    display(ui)
    return ui

