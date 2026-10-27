"""The simulated user (local_llm/user_sim.py): its prompt and how its reply is read (no model)."""

from local_llm import user_sim


def test_prompt_lists_the_conversation_and_the_goal():
    text = user_sim.prompt("想知道內湖天氣", [("今天天氣如何？", "你在哪裡？")], english=False)
    assert "你的目標：想知道內湖天氣" in text and "你：今天天氣如何？" in text
    assert "助理：你在哪裡？" in text and text.endswith("你的下一句：")
    assert "say the first line" in user_sim.prompt("weather", [], english=True)


def test_reply_parsing_reads_json_or_plain_text():
    assert user_sim.parse('{"say": "我在內湖。", "done": false}') == {
        "text": "我在內湖。",
        "done": False,
    }
    assert user_sim.parse('好的：\n{"say": "謝謝，再見。", "done": true}')["done"] is True
    assert user_sim.parse("我在內湖。") == {"text": "我在內湖。", "done": False}
    assert user_sim.parse('{"say": ""}')["done"] is True  # nothing left to say ends it
