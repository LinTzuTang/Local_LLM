"""Live voice assistant: a GPT-Live-style conversation, run as a local cascade.

The header is drawn here, the rest by voice_panel (shared with the scenario gallery); the
conversation runs in the model gateway (/v1/voice/agent/ws). A finished session is
saved as runs/voice_agent-*.json and shown below (and in History).
"""

import ui
import voice_assistant as va
import voice_panel
from i18n import tr

ui.page_header(
    "support_agent",
    tr("即時語音助理", "Live voice assistant"),
    tr(
        "像跟人講電話一樣跟助理對話：它邊聽邊辨識，你一說完就先用一句話回應（例如「好，我幫你查一下內湖的天氣」），"
        "需要查資料時把工具放到背景執行、同時繼續聽你說話，查到後再把結果念給你聽；你隨時可以插話打斷它。"
        f'這個流程參考 OpenAI 的 <a href="{va.GPT_LIVE_URL}" target="_blank">GPT-Live</a>'
        "（快速的語音層維持對話、較重的工作在背景進行）。",
        "Talk to the assistant as you would on the phone: it transcribes while you speak, "
        'answers with a short line the moment you finish ("OK, let me check the weather in '
        'Neihu"), runs lookups in the background while it keeps listening, and speaks the '
        "result when it arrives; you can interrupt it at any time. The flow follows OpenAI's "
        f'<a href="{va.GPT_LIVE_URL}" target="_blank">GPT-Live</a> (a fast voice layer keeps '
        "the conversation going while heavier work runs in the background).",
    ),
    eyebrow=tr("多 Agent 場景", "Scenarios"),
)
voice_panel.panel()
