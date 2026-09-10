"""文件化 Skill 包的加载与校验。"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class SkillPackage:
    name: str
    root: Path
    instructions: str
    prompt: str


@lru_cache(maxsize=32)
def load_skill_package(root: str | Path) -> SkillPackage:
    package_root = Path(root).resolve()
    instruction_path = package_root / "SKILL.md"
    prompt_path = package_root / "prompt.md"
    missing = [path.name for path in (instruction_path, prompt_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Skill 包 {package_root.name} 缺少文件：{', '.join(missing)}")

    instructions = instruction_path.read_text(encoding="utf-8").strip()
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not instructions or not prompt:
        raise ValueError(f"Skill 包 {package_root.name} 的 SKILL.md 或 prompt.md 不能为空")
    return SkillPackage(
        name=package_root.name,
        root=package_root,
        instructions=instructions,
        prompt=prompt,
    )
