"""The home page's description of the machine (local_llm.machine)."""

from local_llm import machine


def test_parse_hardware():
    text = """Hardware:

    Hardware Overview:

      Model Name: MacBook Pro
      Chip: Apple M5 Pro
      Total Number of Cores: 18 (6 performance and 12 efficiency)
      Memory: 64 GB
"""
    hw = machine.parse_hardware(text)
    assert hw["Model Name"] == "MacBook Pro" and hw["Chip"] == "Apple M5 Pro"
    assert hw["Memory"] == "64 GB" and "Hardware" not in hw  # headings have no value


def test_parse_gpus():
    text = "NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887\nOdd GPU, [N/A]\n"
    assert machine.parse_gpus(text) == [
        "NVIDIA RTX PRO 6000 Blackwell Server Edition · 96 GB",
        "Odd GPU",
    ]
    assert machine.parse_gpus("") == []


def test_gpu_memory_sums_gpus_and_processes(monkeypatch):
    def fake_run(*cmd, timeout=5):
        if "--query-gpu=memory.used,memory.total" in cmd:
            return "1024, 97887\n2048, 97887\n"
        return "4242, 30720\n4242, 1024\n77, 512\n"

    monkeypatch.setattr(machine, "_run", fake_run)
    used, total = machine.gpu_memory()
    assert used == 3 and round(total) == 191
    assert machine.gpu_memory_by_pid() == {4242: 31.0, 77: 0.5}


def test_no_nvidia_gpu(monkeypatch):
    monkeypatch.setattr(machine, "_run", lambda *cmd, timeout=5: "")
    assert machine.gpu_memory() is None and machine.gpu_memory_by_pid() == {}
