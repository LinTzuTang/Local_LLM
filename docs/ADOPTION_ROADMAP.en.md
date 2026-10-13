# Simulated adoption: an on-premises AI roadmap for 示範精密工業 (fictional)

> The scenario is fictional: a mid-sized Taiwanese maker and trader of precision components, about 300 people, with Sales, Production, R&D, Quality, Purchasing, Warehouse, Customer Support, HR, Finance and IT. Company data is simulated by the PostgreSQL database that `./scripts/start_db.sh` creates.
>
> Compiled from public sources as of 2026-10-12. Items marked "to verify" rest on secondary sources only; check the original documents before relying on them.

## Why on-premises

- **Data stays in the company**: customer lists, quotes, salaries and ticket contents cannot go to a public-cloud AI.
- **Predictable cost**: a one-off hardware cost instead of per-token billing.
- **Compliance**: meets Taiwan's Personal Data Protection Act, customer NDAs and internal security policy.

The price: you maintain the hardware, model updates, evaluation and security yourself, and the hardware caps the achievable quality.

## Eight phases

| Phase | When | Goal | Deliverables | KPIs | In the platform (✅ done / 🔜 to do) |
| --- | --- | --- | --- | --- | --- |
| **1. Strategy and governance** | Months 0–1 | Set up an AI steering group (GM, IT, legal / HR, champions from each department); write an AI acceptable-use policy (AUP) | Charter, RACI, risk register | Policy signed off; every employee acknowledges the AUP | — |
| **2. Use-case inventory and risk tiers** | Months 1–2 | Workshops per department; rank by value × feasibility × risk | About 15 use cases, 3 chosen as pilots | Estimated hours saved per use case | — |
| **3. Data inventory and classification** (alongside 2) | Months 1–2 | Classify data as public / internal / confidential / personal; map who may see what | Data map, permission matrix | ≥90% of pilot data labelled with an owner | ✅ Department roles and column permissions in the sample database |
| **4. Platform MVP** | Months 2–3 | Local models, agents, UI, access control, audit log | Reference architecture, threat model | p95 latency < 5 s; zero cross-department leaks in red-team tests | ✅ Ollama, model library, chat, agents, external-tool approval, database roles　🔜 Login (SSO), RAG document permissions |
| **5. Evaluation** (before the pilots) | Month 3 | 100–300 Traditional Chinese reference questions per use case, plus attack prompts (prompt injection in documents, personal-data extraction) | Automated regression evaluation | ≥85% correct on grounded answers; injection success < 2% | ✅ Model comparison, LLM judge, traces　🔜 Evaluation-set page |
| **6. Controlled pilot** | Months 3–6, about 30 people | Three pilots go live; agents read only, and every write needs human approval | Weekly feedback, incident log | ≥60% weekly active; support handling time −20%; < 1 escalation per 100 conversations | ✅ Read-only database, external-call approval　🔜 Feedback button, usage quotas |
| **7. Harden and scale** | Months 6–9 | Pick hardware for the concurrency you need (DGX Spark or a GPU server); move to vLLM, add redundancy, AD account sync, DLP | Production environment | Cost per answer, GPU utilisation, availability ≥99% | 🔜 vLLM (Spark), multi-model routing |
| **8. Ongoing operations** | Ongoing | Drive change (champions, training, sharing wins), quarterly risk review, run the evaluation before every model update | Quarterly report, audit log | Adoption per department, hours saved, audit findings closed | ✅ Run history (runs/)　🔜 Usage dashboard |

### Three suggested pilots (all can be demoed on this platform)

| Pilot | Department | Platform feature | Sample questions |
| --- | --- | --- | --- |
| Sales / inventory assistant | Sales | Company database: Sales assistant (read-only) | "Which city had the highest revenue in 2026?" "Which products are below safety stock?" |
| Support ticket assistant | Customer Support | Company database: Support assistant | "Which urgent tickets are still open?" |
| HR policy Q&A | HR | 🔜 RAG (leave, overtime and travel policies) + HR assistant | "How is annual leave calculated?" "Average monthly salary per department" |

## Permission design: what this platform demonstrates

**Keeping AI away from data it should not see cannot rely on the prompt alone.**

| Control layer | How the platform does it | How to see it |
| --- | --- | --- |
| Database roles | Each agent connects with its own department's PostgreSQL role; columns such as salary and cost are granted only to specific roles | Ask the Sales assistant for "each person's monthly salary" → it answers that it lacks permission |
| Read-only + timeout | Roles have SELECT only, transactions default to read-only, queries time out after 5 s | The database refuses writes and long-running queries |
| Tool visibility | With network access off, external tools are never given to the model | The "Allow network tools" switch in the Agent sidebar |
| Human approval | Pauses before every external call, shows the arguments about to be sent, and lets you approve, edit or reject | The research assistant looking up weather or Wikipedia |
| Call limits | At most N calls per external tool per task (enforced in code) | Sidebar setting |
| Intranet guard | The web-fetch tool blocks localhost, private networks and cloud metadata addresses | Stops an injected URL from reaching internal systems |
| Audit | Every run is saved as JSON: prompt, each LLM call, tool arguments and results, approval decisions | The "📜 Run history" page |

**Not done yet; required before production:**
- User login and group permissions. Roles currently follow the chosen agent, not the signed-in user.
- Permission filtering for RAG documents. It must happen at retrieval time, not just be hidden in the UI.
- Defences and tests against prompt injection in documents and tool output.
- Usage quotas and a dashboard.

## Common pitfalls

1. **RAG leaks**: without passage-level access control, someone who should not see salary or personnel documents can get them by asking.
2. **Indirect prompt injection**: malicious instructions hidden in an email, PDF or web page reach the model through tool results.
3. **Giving agents write access**: placing orders or changing the ERP must always require human approval.
4. **No evaluation set**: going live because it "feels fine" leaves you nothing to compare against when you switch models.
5. **Misjudging hardware**: concurrent users, not model size, drive the hardware requirement. A 30B model at 4-bit is about 20GB, plus the KV cache.
6. **Mixed Traditional / Simplified Chinese**: model output slips into Simplified characters; evaluate for it and constrain it in the prompt.
7. **Shadow AI**: employees quietly use public-cloud AI; offer a good internal alternative backed by policy.
8. **Pilot scope too large**: make it genuinely useful for one department first, then expand.

## Open-source platforms worth studying (borrow the architecture; adoption optional)

| Platform | License (checked 2026-10) | Worth borrowing |
| --- | --- | --- |
| Open WebUI | BSD-3 plus a branding clause: with more than 50 users in 30 days you may not remove the branding unless you buy an enterprise license | Group permissions, per-group model access, pre/post-processing hooks |
| Dify | Modified Apache 2.0 (not OSI open source): multi-tenant use needs a commercial license | Workspace isolation, versioned workflows, logging and annotation UI |
| RAGFlow | Apache-2.0 | Layout-aware chunking, source citations |
| Onyx (formerly Danswer) | MIT, except the `ee/` directory | Connectors that sync source-system permissions (enterprise feature) |
| AnythingLLM / LibreChat | MIT | Multi-user workspaces, MCP tools, per-user usage limits |
| n8n | Sustainable Use License (not OSI open source) | Approval steps before actions, execution logs |
| LangGraph | MIT (self-hosting the deployment platform needs an enterprise plan) | Checkpoints, `interrupt()` for human approval (used by this platform) |

## Governance frameworks

| Framework | Use |
| --- | --- |
| NIST AI RMF 1.0 + Generative AI Profile (NIST AI 600-1) | Skeleton for the risk register |
| ISO/IEC 42001:2023 | Certifiable AI management system; integrates with existing ISO 9001 / 27001 |
| OWASP Top 10 for LLM Applications, OWASP Top 10 for Agentic Applications | Threat model for prompt injection, excessive agency and more (latest version numbers to verify at genai.owasp.org) |
| Taiwan's AI Basic Act | Per secondary sources: passed third reading 2025-12, in force 2026-01; a framework law overseen by the National Science and Technology Council (to verify) |
| Ministry of Digital Affairs AI risk classification framework | Risk-tiering use cases (per secondary sources; to verify at moda.gov.tw) |
| Executive Yuan guidelines for government agencies using generative AI | Apply to the public sector; a usable template for a corporate AUP: no confidential data in external generative AI, and a human makes the final decision |
| Financial Supervisory Commission guidelines for AI in the financial industry | Not binding outside finance, but useful as a control checklist |
| Personal Data Protection Act | Legal basis when running RAG over HR and customer data |

## Main sources

- Datasets: EnterpriseBench (MIT) https://github.com/ast-fri/EnterpriseBench; AdventureWorks for Postgres (MIT) https://github.com/lorint/AdventureWorks-for-Postgres; Faker (MIT) https://github.com/joke2k/faker
- Platform licenses: https://docs.openwebui.com/license/; https://github.com/langgenius/dify; https://github.com/infiniflow/ragflow; https://docs.onyx.app
- Governance: NIST AI 600-1 https://www.nist.gov/itl/ai-risk-management-framework; OWASP GenAI https://genai.owasp.org; coverage of Taiwan's AI Basic Act https://www.leeandli.com/EN/NewslettersDetail/7566.htm
