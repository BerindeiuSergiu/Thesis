from __future__ import annotations

from plotting_common import write_markdown


content = """
# FIGURA 1.1 - Contributiile principale ale lucrarii

```mermaid
flowchart LR
    A[Video indoor local] --> B[Selectie reprezentativa de cadre]
    B --> C[Fast3R multiview inference]
    C --> D[Procesare comuna a geometriei]
    D --> E[Output Gaussian inspectabil]
    D --> F[Output mesh exportabil]
    C --> G[Evaluare runtime si limite de memorie]
    C --> H[Experiment head-only fine-tuning pe ARKitScenes]
```

Aceasta figura este conceptuala si poate fi recreata in Word/PowerPoint dupa schema Mermaid.
Contributia centrala care trebuie vizualizata este reducerea timpului pana la o scena inspectabila,
nu doar existenta unei reconstructii 3D.
"""


if __name__ == "__main__":
    path = write_markdown(
        "figura_1_1_contributiile_principale_ale_lucrarii.md",
        content,
    )
    print(path)
