"""
app.py
Streamlit dashboard for the Intelligent Supplier Risk Intelligence System.
"""

import os
import tempfile
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from src.ingestion import ingest_supplier_pdf, DocumentChunk
from src.risk_extractor import analyze_all_chunks, SupplierRiskProfile
from src.rag_pipeline import SupplierVectorStore, SupplierRAGAssistant

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Supplier Risk Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "risk_profiles" not in st.session_state:
    st.session_state.risk_profiles = {}
if "vector_store" not in st.session_state:
    st.session_state.vector_store = SupplierVectorStore()
if "rag_assistant" not in st.session_state:
    st.session_state.rag_assistant = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/supply-chain.png", width=60)
    st.title("⚙️ Configuration")

    st.markdown("---")
    st.subheader("📄 Upload Supplier PDFs")
    uploaded_files = st.file_uploader(
        "Upload one or more supplier documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    openai_key = st.text_input(
        "🔑 OpenAI API Key (optional)",
        type="password",
        placeholder="sk-... (leave blank for local mode)"
    )
    st.caption("Without an API key, the system uses local keyword-based retrieval.")

    process_btn = st.button("🚀 Process Documents", use_container_width=True, type="primary")

    if process_btn and uploaded_files:
        all_chunks = []
        with st.spinner("Extracting text and analyzing risks..."):
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                supplier_name = os.path.splitext(uploaded_file.name)[0]
                chunks = ingest_supplier_pdf(tmp_path, supplier_name=supplier_name)
                all_chunks.extend(chunks)
                os.unlink(tmp_path)

        st.session_state.chunks = all_chunks
        st.session_state.risk_profiles = analyze_all_chunks(all_chunks)

        with st.spinner("Building vector index..."):
            vs = SupplierVectorStore()
            vs.build_index(all_chunks)
            st.session_state.vector_store = vs
            st.session_state.rag_assistant = SupplierRAGAssistant(
                vs, openai_api_key=openai_key if openai_key else None
            )

        st.success(f"✅ Processed {len(uploaded_files)} document(s), {len(all_chunks)} chunks indexed!")

    st.markdown("---")
    st.caption("Built with LangChain · FAISS · spaCy · Streamlit")


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.title("🔍 Intelligent Supplier Risk Intelligence System")
st.markdown("*NLP + RAG-powered supply chain risk monitoring*")

tab1, tab2, tab3 = st.tabs(["📊 Risk Dashboard", "💬 AI Q&A Assistant", "📋 Signal Details"])


# ── Tab 1: Risk Dashboard ──────────────────────────────────────────────────
with tab1:
    if not st.session_state.risk_profiles:
        st.info("👈 Upload supplier PDF documents in the sidebar to get started.")

        # Show demo layout
        st.markdown("### What this system does:")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📂 Document Ingestion", "PDF → Chunks", "PyMuPDF")
        col2.metric("🧠 NER & NLP", "Risk Signals", "spaCy + Keywords")
        col3.metric("🗄️ Vector Search", "FAISS Index", "sentence-transformers")
        col4.metric("🤖 RAG Q&A", "Contextual AI", "LangChain + GPT")
    else:
        profiles = st.session_state.risk_profiles

        # Summary metrics
        st.subheader("📈 Portfolio Risk Overview")
        total = len(profiles)
        high_risk = sum(1 for p in profiles.values() if p.risk_level == "High")
        med_risk = sum(1 for p in profiles.values() if p.risk_level == "Medium")
        low_risk = sum(1 for p in profiles.values() if p.risk_level == "Low")
        avg_score = round(sum(p.overall_score for p in profiles.values()) / total, 1)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Suppliers", total)
        c2.metric("🔴 High Risk", high_risk)
        c3.metric("🟡 Medium Risk", med_risk)
        c4.metric("🟢 Low Risk", low_risk)
        c5.metric("Avg Risk Score", f"{avg_score}/100")

        st.markdown("---")

        # Risk score table
        st.subheader("🏭 Supplier Risk Scores")
        rows = []
        for name, p in profiles.items():
            rows.append({
                "Supplier": name,
                "Overall Score": p.overall_score,
                "Geopolitical": p.geopolitical_score,
                "Financial": p.financial_score,
                "Compliance": p.compliance_score,
                "Operational": p.operational_score,
                "Risk Level": p.risk_level
            })

        df = pd.DataFrame(rows).sort_values("Overall Score", ascending=False)

        def color_risk(val):
            if val == "High":
                return "background-color: #ffcccc"
            elif val == "Medium":
                return "background-color: #fff3cc"
            return "background-color: #ccffcc"

        st.dataframe(
            df.style.applymap(color_risk, subset=["Risk Level"]),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")

        # Charts
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Risk Score Breakdown")
            fig_bar = go.Figure()
            categories = ["Geopolitical", "Financial", "Compliance", "Operational"]
            colors = ["#e74c3c", "#f39c12", "#9b59b6", "#3498db"]

            for cat, color in zip(categories, colors):
                fig_bar.add_trace(go.Bar(
                    name=cat,
                    x=df["Supplier"],
                    y=df[cat],
                    marker_color=color
                ))

            fig_bar.update_layout(
                barmode="group",
                xaxis_title="Supplier",
                yaxis_title="Risk Score (0–100)",
                legend_title="Risk Category",
                height=350,
                margin=dict(t=20)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_right:
            st.subheader("Risk Level Distribution")
            risk_counts = df["Risk Level"].value_counts().reset_index()
            risk_counts.columns = ["Risk Level", "Count"]
            fig_pie = px.pie(
                risk_counts,
                names="Risk Level",
                values="Count",
                color="Risk Level",
                color_discrete_map={"High": "#e74c3c", "Medium": "#f39c12", "Low": "#2ecc71"},
                height=350
            )
            fig_pie.update_layout(margin=dict(t=20))
            st.plotly_chart(fig_pie, use_container_width=True)

        # Radar chart for selected supplier
        st.markdown("---")
        st.subheader("🎯 Supplier Risk Radar")
        selected = st.selectbox("Select Supplier", list(profiles.keys()))
        p = profiles[selected]

        fig_radar = go.Figure(go.Scatterpolar(
            r=[p.geopolitical_score, p.financial_score, p.compliance_score, p.operational_score, p.geopolitical_score],
            theta=["Geopolitical", "Financial", "Compliance", "Operational", "Geopolitical"],
            fill="toself",
            fillcolor="rgba(231, 76, 60, 0.2)",
            line=dict(color="#e74c3c")
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            height=400,
            margin=dict(t=30)
        )
        st.plotly_chart(fig_radar, use_container_width=True)


# ── Tab 2: RAG Q&A ────────────────────────────────────────────────────────
with tab2:
    st.subheader("💬 Ask the Supplier Risk Assistant")

    if not st.session_state.rag_assistant:
        st.info("👈 Process supplier documents first to enable the Q&A assistant.")
    else:
        supplier_options = ["All Suppliers"] + list(st.session_state.risk_profiles.keys())
        selected_supplier = st.selectbox("Filter by supplier (optional)", supplier_options)
        supplier_filter = None if selected_supplier == "All Suppliers" else selected_supplier

        example_questions = [
            "What financial risks are mentioned in the documents?",
            "Are there any compliance violations or regulatory issues?",
            "What geopolitical risks affect this supplier?",
            "Are there any signs of supply chain disruptions?",
            "Summarize the key risks for this supplier."
        ]
        st.caption("💡 Try: " + " | ".join(f'"{q}"' for q in example_questions[:3]))

        question = st.text_input("Your question:", placeholder="e.g. What financial risks are mentioned?")

        if st.button("🔎 Ask", type="primary") and question:
            with st.spinner("Retrieving relevant documents and generating answer..."):
                result = st.session_state.rag_assistant.ask(
                    question,
                    supplier_filter=supplier_filter
                )

            st.session_state.chat_history.append({
                "question": question,
                "answer": result["answer"],
                "sources": result["sources"]
            })

        # Display chat history
        if st.session_state.chat_history:
            st.markdown("---")
            for item in reversed(st.session_state.chat_history):
                with st.chat_message("user"):
                    st.write(item["question"])
                with st.chat_message("assistant"):
                    st.write(item["answer"])
                    if item["sources"]:
                        st.caption(f"📎 Sources: {', '.join(item['sources'])}")

            if st.button("🗑️ Clear History"):
                st.session_state.chat_history = []
                st.rerun()


# ── Tab 3: Signal Details ──────────────────────────────────────────────────
with tab3:
    st.subheader("📋 Extracted Risk Signals")

    if not st.session_state.risk_profiles:
        st.info("Process documents to view extracted risk signals.")
    else:
        supplier_sel = st.selectbox("Select Supplier", list(st.session_state.risk_profiles.keys()), key="sig_sel")
        profile = st.session_state.risk_profiles[supplier_sel]

        if not profile.risk_signals:
            st.success("No significant risk signals detected for this supplier.")
        else:
            type_filter = st.multiselect(
                "Filter by risk type",
                ["geopolitical", "financial", "compliance", "operational"],
                default=["geopolitical", "financial", "compliance", "operational"]
            )

            for signal in profile.risk_signals:
                if signal.risk_type not in type_filter:
                    continue

                color_map = {
                    "high": "🔴",
                    "medium": "🟡",
                    "low": "🟢"
                }
                emoji = color_map.get(signal.severity, "⚪")

                with st.expander(f"{emoji} [{signal.risk_type.upper()}] — Severity: {signal.severity.capitalize()} | Chunk: {signal.chunk_id}"):
                    st.markdown(f"**Keywords found:** `{'`, `'.join(signal.matched_keywords)}`")
                    st.markdown(f"**Context:**\n> {signal.context_snippet}")
