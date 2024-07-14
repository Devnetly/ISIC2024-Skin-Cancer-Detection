from typing import Any

class BatchResults:

    def __init__(self, names : list[str]) -> None:

        self.names = names
        self.values = { name: [] for name in names }

    def update(self, name : str, value : Any) -> None:
        self.values[name].append(value)

    def update_all(self, values : dict[str, Any]) -> None:
        for name, value in values.items():
            self.update(name, value)

    def compute(self) -> dict[str, Any]:
        
        results = {}

        for name, values in self.values.items():
            results[name] = sum(values) / len(values)

        return results