---
name: internal-comms
description: A set of resources to help me write all kinds of internal communications, using the formats
  that my company likes to use. Claude should use this skill whenever asked to write some sort of internal
  communications (status reports, leadership updates, 3P updates, company newsletters, FAQs, incident
  reports, project updates, etc.).
description_zh: 依公司偏好格式撰寫各種內部溝通：3P 週報（進度／計畫／問題）、主管更新、公司電子報、FAQ 回覆、狀態報告、事故報告、專案更新。要寫內部溝通文件時使用（範例檔在 examples/）。
source: https://github.com/anthropics/skills/tree/dbd4588f9e1033efb41dad4bef2f7947c8993d44/skills/internal-comms
license: Apache-2.0
modified: Local Agent Lab added description_zh, source and license to this front matter (licence text
  in LICENSE.txt); the instructions below are unchanged.
---
## When to use this skill
To write internal communications, use this skill for:
- 3P updates (Progress, Plans, Problems)
- Company newsletters
- FAQ responses
- Status reports
- Leadership updates
- Project updates
- Incident reports

## How to use this skill

To write any internal communication:

1. **Identify the communication type** from the request
2. **Load the appropriate guideline file** from the `examples/` directory:
    - `examples/3p-updates.md` - For Progress/Plans/Problems team updates
    - `examples/company-newsletter.md` - For company-wide newsletters
    - `examples/faq-answers.md` - For answering frequently asked questions
    - `examples/general-comms.md` - For anything else that doesn't explicitly match one of the above
3. **Follow the specific instructions** in that file for formatting, tone, and content gathering

If the communication type doesn't match any existing guideline, ask for clarification or more context about the desired format.

## Keywords
3P updates, company newsletter, company comms, weekly update, faqs, common questions, updates, internal comms
