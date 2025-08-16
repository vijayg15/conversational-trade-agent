from typing import List, Dict

class Memory:
    def __init__(self, max_turns=8):
        self.turns: List[Dict[str,str]] = []
        self._max = max_turns
    def add(self, role: str, text: str):
        self.turns.append({"role": role, "text": text})
        self.turns = self.turns[-2*self._max:]
    def last_context(self, n=4) -> str:
        return "\\n".join([f"{t['role']}: {t['text']}" for t in self.turns[-n:]])