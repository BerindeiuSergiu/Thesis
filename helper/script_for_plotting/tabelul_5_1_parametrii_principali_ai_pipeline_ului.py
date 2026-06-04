from __future__ import annotations

from pathlib import Path

from plotting_common import markdown_table, repo_root, write_markdown


def parse_fast3r_settings(text: str) -> dict[str, object]:
    values: dict[str, object] = {}
    in_fast3r = False
    base_indent = None
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if stripped == "fast3r:":
            in_fast3r = True
            base_indent = indent
            continue
        if in_fast3r and base_indent is not None and indent <= base_indent:
            break
        if not in_fast3r or ":" not in stripped:
            continue
        key, raw_value = stripped.split(":", 1)
        raw_value = raw_value.strip()
        if raw_value == "":
            continue
        if raw_value.startswith('"') and raw_value.endswith('"'):
            value: object = raw_value[1:-1]
        elif raw_value.lower() == "true":
            value = True
        elif raw_value.lower() == "false":
            value = False
        else:
            try:
                value = int(raw_value)
            except ValueError:
                try:
                    value = float(raw_value)
                except ValueError:
                    value = raw_value
        values.setdefault(key, value)
    return values


def main() -> None:
    config_path = repo_root() / "src/config/settings.yaml"
    fast3r = parse_fast3r_settings(config_path.read_text(encoding="utf-8"))

    keys = [
        ("active_preset", fast3r.get("active_preset"), "Presetul folosit de pipeline"),
        ("target_frames", fast3r.get("target_frames"), "Numarul tinta de cadre selectate"),
        ("selection_mode", fast3r.get("selection_mode"), "Strategia de selectie a cadrelor"),
        ("visual_shortlist_target", fast3r.get("visual_shortlist_target"), "Shortlist vizual inainte de proba geometrica"),
        ("probe_max_frames", fast3r.get("probe_max_frames"), "Numar maxim de cadre in proba geometrica"),
        ("fast3r_image_size", fast3r.get("fast3r_image_size"), "Rezolutia de intrare Fast3R"),
        ("dtype", fast3r.get("dtype"), "Tip numeric pentru memorie/performanta"),
        ("min_confidence_threshold", fast3r.get("min_confidence_threshold"), "Filtrare dupa increderea modelului"),
        ("confidence_keep_ratio", fast3r.get("confidence_keep_ratio"), "Retentie dupa incredere"),
        ("voxel_size", fast3r.get("voxel_size"), "Downsampling spatial"),
        ("scale_reference_real", fast3r.get("scale_reference_real"), "Referinta de scala in metri"),
        ("gaussian_enabled", fast3r.get("gaussian_enabled"), "Activarea ramurii Gaussian"),
        ("mesh_enabled", fast3r.get("mesh_enabled"), "Activarea ramurii mesh in presetul curent"),
    ]

    content = "# TABELUL 5.1 - Parametrii principali ai pipeline-ului\n\n"
    content += markdown_table(["Parametru", "Valoare", "Rol in pipeline"], keys)
    content += f"\n\nSursa: `{Path('src/config/settings.yaml')}`"
    path = write_markdown("tabelul_5_1_parametrii_principali_ai_pipeline_ului.md", content)
    print(path)


if __name__ == "__main__":
    main()
