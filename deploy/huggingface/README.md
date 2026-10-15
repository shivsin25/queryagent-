---
title: Query Tool
emoji: 📚
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
pinned: false
short_description: Grounded, cited answers from your own documents
---

# Query Tool

A self-correcting, multi-agent RAG system. Agents plan the search, find the relevant passages,
draft an answer that cites a source for every fact, and send it to a reviewer and a citation
checker before you see it. If the documents don't contain the answer, it says so.

The demo knowledge base describes a fictional company, Nimbus Analytics: its pricing,
security policy, support SLA and a data processing agreement (PDF). Upload your own files on
the **Knowledge base** page. Uploads last until the Space restarts.

Source code and documentation: https://github.com/shivsin25/queryagent-

## Configuration (Space settings → Variables and secrets)

| Name | Kind | Example |
|---|---|---|
| `QT_LLM_PROVIDER` | variable | `openai`, `gemini` or `groq` |
| `OPENAI_API_KEY` / `GOOGLE_API_KEY` / `GROQ_API_KEY` | secret | the key for each provider you use |
| `QT_REVIEWER_PROVIDER` | variable | `gemini`, to review on a different provider |
| `QT_DAILY_BUDGET_USD` | variable | `0.50`, the cap on LLM spend per day |
| `QT_UI_PASSWORD` | secret | if set, visitors need this password to use the app |
