"""Heuristic detection of language, project type, framework, and key files.

Uses filenames and lightweight manifest parsing. No LLM involved.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path

from agent.explorer.models import ImportantFile, RepositoryMap

logger = logging.getLogger(__name__)

IMPORTANT_FILE_RULES: list[tuple[str, str]] = [
    ("package.json", "manifest"),
    ("package-lock.json", "lockfile"),
    ("yarn.lock", "lockfile"),
    ("pnpm-lock.yaml", "lockfile"),
    ("requirements.txt", "manifest"),
    ("pyproject.toml", "manifest"),
    ("setup.py", "manifest"),
    ("setup.cfg", "manifest"),
    ("Pipfile", "manifest"),
    ("poetry.lock", "lockfile"),
    ("go.mod", "manifest"),
    ("go.sum", "lockfile"),
    ("Cargo.toml", "manifest"),
    ("Cargo.lock", "lockfile"),
    ("pom.xml", "manifest"),
    ("build.gradle", "manifest"),
    ("build.gradle.kts", "manifest"),
    ("settings.gradle", "config"),
    ("settings.gradle.kts", "config"),
    ("Dockerfile", "container"),
    ("docker-compose.yml", "container"),
    ("docker-compose.yaml", "container"),
    ("tsconfig.json", "config"),
    ("Makefile", "build"),
    ("Gemfile", "manifest"),
    ("composer.json", "manifest"),
]

README_NAMES = frozenset({"readme.md", "readme.rst", "readme.txt", "readme"})

IMPORTANT_DIRECTORY_NAMES = frozenset(
    {
        "src",
        "app",
        "lib",
        "config",
        "configs",
        "cmd",
        "pkg",
        "internal",
        "tests",
        "test",
        "controllers",
        "models",
        "routes",
        "services",
        "api",
        "bin",
        "scripts",
        "docs",
    }
)

# Relative path candidates commonly used as entry points (ordered by preference within a stack).
NODE_ENTRY_CANDIDATES = (
    "server.js",
    "server.ts",
    "index.js",
    "index.ts",
    "app.js",
    "app.ts",
    "src/index.js",
    "src/index.ts",
    "src/server.js",
    "src/server.ts",
    "src/main.js",
    "src/main.ts",
    "main.js",
)

PYTHON_ENTRY_CANDIDATES = (
    "main.py",
    "app.py",
    "manage.py",
    "wsgi.py",
    "asgi.py",
    "src/main.py",
    "src/app.py",
    "__main__.py",
)

GO_ENTRY_CANDIDATES = (
    "main.go",
    "cmd/main.go",
)

RUST_ENTRY_CANDIDATES = (
    "src/main.rs",
    "main.rs",
)

JAVA_ENTRY_HINTS = (
    "src/main/java",
)


def content_by_basename(file_contents: dict[str, str], basename: str) -> str | None:
    """Return content for a file whose path equals or ends with ``basename``."""
    if basename in file_contents:
        return file_contents[basename]
    matches = [
        (path, content)
        for path, content in file_contents.items()
        if Path(path).name == basename
    ]
    if not matches:
        return None
    matches.sort(key=lambda item: (item[0].count("/"), len(item[0])))
    return matches[0][1]


class ProjectDetector:
    """Infer project metadata from a RepositoryMap and selective file reads."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()

    def detect_important_file_paths(self, repo_map: RepositoryMap) -> list[tuple[str, str]]:
        """Return ``(relative_path, role)`` for important files present in the map."""
        found: list[tuple[str, str]] = []
        seen: set[str] = set()

        for basename, role in IMPORTANT_FILE_RULES:
            root_candidate = basename
            if root_candidate in repo_map.files and root_candidate not in seen:
                found.append((root_candidate, role))
                seen.add(root_candidate)
                continue

            matches = [
                f
                for f in repo_map.files
                if Path(f).name.lower() == basename.lower() and f not in seen
            ]
            if matches:
                matches.sort(key=lambda p: (p.count("/"), len(p)))
                chosen = matches[0]
                found.append((chosen, role))
                seen.add(chosen)

        readme_matches = [
            f
            for f in repo_map.files
            if Path(f).name.lower() in README_NAMES and f not in seen
        ]
        if readme_matches:
            readme_matches.sort(key=lambda p: (p.count("/"), len(p)))
            chosen = readme_matches[0]
            found.append((chosen, "readme"))
            seen.add(chosen)

        logger.info("Detected %d important file candidates", len(found))
        return found

    def detect_important_directories(self, repo_map: RepositoryMap) -> list[str]:
        """Return important directory paths (relative), shallowest-first."""
        important: list[str] = []
        for directory in repo_map.directories:
            name = Path(directory).name.lower()
            if name in IMPORTANT_DIRECTORY_NAMES:
                important.append(directory)
        important.sort(key=lambda p: (p.count("/"), p.lower()))
        return important

    def detect_project_name(self, repo_map: RepositoryMap, file_contents: dict[str, str]) -> str:
        """Infer project name from manifests or directory name."""
        pkg = content_by_basename(file_contents, "package.json")
        if pkg:
            try:
                data = json.loads(pkg)
                name = data.get("name")
                if isinstance(name, str) and name.strip():
                    return name.strip()
            except json.JSONDecodeError:
                logger.debug("package.json is not valid JSON")

        pyproject = content_by_basename(file_contents, "pyproject.toml")
        if pyproject:
            match = re.search(
                r'(?m)^\s*name\s*=\s*["\']([^"\']+)["\']',
                pyproject,
            )
            if match:
                return match.group(1)

        cargo = content_by_basename(file_contents, "Cargo.toml")
        if cargo:
            match = re.search(
                r'(?m)^\s*name\s*=\s*["\']([^"\']+)["\']',
                cargo,
            )
            if match:
                return match.group(1)

        go_mod = content_by_basename(file_contents, "go.mod")
        if go_mod:
            match = re.search(r"(?m)^\s*module\s+(\S+)", go_mod)
            if match:
                module = match.group(1)
                return module.rstrip("/").split("/")[-1]

        return Path(repo_map.root).name

    def detect_language_and_type(
        self,
        repo_map: RepositoryMap,
        file_contents: dict[str, str],
    ) -> tuple[str, str]:
        """Return ``(detected_language, project_type)``.

        ``project_type`` is one of: Node.js, Python, Java, Go, Rust, or Unknown.
        """
        files = set(repo_map.files)
        basenames = {Path(f).name for f in files}

        scores: Counter[str] = Counter()

        if "package.json" in files or "package.json" in basenames:
            scores["Node.js"] += 5
        if any(name in files or name in basenames for name in ("requirements.txt", "pyproject.toml", "setup.py", "Pipfile")):
            scores["Python"] += 5
        if "go.mod" in files or "go.mod" in basenames:
            scores["Go"] += 5
        if "Cargo.toml" in files or "Cargo.toml" in basenames:
            scores["Rust"] += 5
        if any(name in files or name in basenames for name in ("pom.xml", "build.gradle", "build.gradle.kts")):
            scores["Java"] += 5

        ext_counts: Counter[str] = Counter()
        for path in repo_map.files:
            ext_counts[Path(path).suffix.lower()] += 1

        if ext_counts[".js"] + ext_counts[".ts"] + ext_counts[".jsx"] + ext_counts[".tsx"]:
            scores["Node.js"] += min(3, (ext_counts[".js"] + ext_counts[".ts"]) // 2)
        if ext_counts[".py"]:
            scores["Python"] += min(3, ext_counts[".py"] // 2)
        if ext_counts[".go"]:
            scores["Go"] += min(3, ext_counts[".go"] // 2)
        if ext_counts[".rs"]:
            scores["Rust"] += min(3, ext_counts[".rs"] // 2)
        if ext_counts[".java"]:
            scores["Java"] += min(3, ext_counts[".java"] // 2)

        if not scores:
            return "Unknown", "Unknown"

        project_type = scores.most_common(1)[0][0]
        language_map = {
            "Node.js": "JavaScript",
            "Python": "Python",
            "Java": "Java",
            "Go": "Go",
            "Rust": "Rust",
        }
        if project_type == "Node.js" and ext_counts[".ts"] + ext_counts[".tsx"] > ext_counts[".js"] + ext_counts[".jsx"]:
            return "TypeScript", "Node.js"

        return language_map.get(project_type, "Unknown"), project_type

    def detect_framework(
        self,
        project_type: str,
        file_contents: dict[str, str],
        repo_map: RepositoryMap,
    ) -> str | None:
        """Best-effort framework detection from manifests and filenames."""
        if project_type == "Node.js":
            return self._detect_node_framework(file_contents)
        if project_type == "Python":
            return self._detect_python_framework(file_contents)
        if project_type == "Java":
            return self._detect_java_framework(file_contents, repo_map)
        if project_type == "Go":
            return self._detect_go_framework(file_contents)
        if project_type == "Rust":
            return self._detect_rust_framework(file_contents)
        return None

    def detect_entry_point(
        self,
        project_type: str,
        repo_map: RepositoryMap,
        file_contents: dict[str, str],
    ) -> str | None:
        """Infer a likely entry point path relative to the repo root."""
        files = set(repo_map.files)

        if project_type == "Node.js":
            pkg = content_by_basename(file_contents, "package.json")
            if pkg:
                try:
                    data = json.loads(pkg)
                    main = data.get("main")
                    if isinstance(main, str) and main.strip():
                        normalized = main.strip().lstrip("./")
                        if normalized in files:
                            return normalized
                        return normalized
                except json.JSONDecodeError:
                    pass
            for candidate in NODE_ENTRY_CANDIDATES:
                if candidate in files:
                    return candidate

        if project_type == "Python":
            for candidate in PYTHON_ENTRY_CANDIDATES:
                if candidate in files:
                    return candidate
            mains = [f for f in files if Path(f).name == "main.py"]
            if mains:
                mains.sort(key=lambda p: (p.count("/"), len(p)))
                return mains[0]

        if project_type == "Go":
            for candidate in GO_ENTRY_CANDIDATES:
                if candidate in files:
                    return candidate
            mains = [f for f in files if Path(f).name == "main.go"]
            if mains:
                mains.sort(key=lambda p: (p.count("/"), len(p)))
                return mains[0]

        if project_type == "Rust":
            for candidate in RUST_ENTRY_CANDIDATES:
                if candidate in files:
                    return candidate

        if project_type == "Java":
            for hint in JAVA_ENTRY_HINTS:
                if any(f.startswith(hint) and f.endswith(".java") for f in files):
                    apps = [
                        f
                        for f in files
                        if f.startswith(hint) and f.endswith("Application.java")
                    ]
                    if apps:
                        apps.sort(key=lambda p: (p.count("/"), len(p)))
                        return apps[0]
                    javas = [f for f in files if f.startswith(hint) and f.endswith(".java")]
                    if javas:
                        javas.sort(key=lambda p: (p.count("/"), len(p)))
                        return javas[0]

        return None

    def build_technology_stack(
        self,
        *,
        language: str,
        project_type: str,
        framework: str | None,
        important_files: list[ImportantFile],
        file_contents: dict[str, str],
    ) -> list[str]:
        """Assemble an ordered list of technology labels."""
        stack: list[str] = []
        if project_type != "Unknown":
            stack.append(project_type)
        if language not in ("Unknown", project_type) and language not in stack:
            stack.append(language)
        if framework and framework not in stack:
            stack.append(framework)

        roles = {item.role for item in important_files}
        paths = {item.path for item in important_files}

        if "Dockerfile" in paths or any(Path(p).name == "Dockerfile" for p in paths):
            stack.append("Docker")
        if any(Path(p).name.startswith("docker-compose") for p in paths):
            stack.append("Docker Compose")

        if project_type == "Node.js":
            pkg = content_by_basename(file_contents, "package.json")
            if pkg:
                try:
                    data = json.loads(pkg)
                    deps = {
                        **(data.get("dependencies") or {}),
                        **(data.get("devDependencies") or {}),
                    }
                    for name, label in (
                        ("mongoose", "Mongoose"),
                        ("mongodb", "MongoDB"),
                        ("express", "Express"),
                        ("react", "React"),
                        ("next", "Next.js"),
                        ("vue", "Vue"),
                        ("typescript", "TypeScript"),
                    ):
                        if name in deps and label not in stack:
                            stack.append(label)
                    if "mongoose" in deps and "MongoDB" not in stack:
                        stack.append("MongoDB")
                except json.JSONDecodeError:
                    pass

        if "lockfile" in roles and "npm" not in stack and project_type == "Node.js":
            stack.append("npm")

        return stack

    def _detect_node_framework(self, file_contents: dict[str, str]) -> str | None:
        pkg = content_by_basename(file_contents, "package.json")
        if not pkg:
            return None
        try:
            data = json.loads(pkg)
        except json.JSONDecodeError:
            return None
        deps = {
            **(data.get("dependencies") or {}),
            **(data.get("devDependencies") or {}),
        }
        checks = (
            ("next", "Next.js"),
            ("@nestjs/core", "NestJS"),
            ("express", "Express"),
            ("fastify", "Fastify"),
            ("koa", "Koa"),
            ("react", "React"),
            ("vue", "Vue"),
            ("@angular/core", "Angular"),
        )
        for key, label in checks:
            if key in deps:
                return label
        return None

    def _detect_python_framework(self, file_contents: dict[str, str]) -> str | None:
        blob = "\n".join(
            content
            for key, content in file_contents.items()
            if key.endswith(("requirements.txt", "pyproject.toml", "Pipfile", "setup.py"))
            or Path(key).name
            in {"requirements.txt", "pyproject.toml", "Pipfile", "setup.py"}
        ).lower()
        combined = "\n".join(file_contents.values()).lower()
        text = blob or combined
        checks = (
            ("django", "Django"),
            ("fastapi", "FastAPI"),
            ("flask", "Flask"),
            ("tornado", "Tornado"),
        )
        for needle, label in checks:
            if needle in text:
                return label
        return None

    def _detect_java_framework(
        self,
        file_contents: dict[str, str],
        repo_map: RepositoryMap,
    ) -> str | None:
        combined = "\n".join(file_contents.values()).lower()
        if "spring-boot" in combined or "springframework.boot" in combined:
            return "Spring Boot"
        if any(Path(f).name == "pom.xml" for f in repo_map.files):
            return "Maven"
        if any(Path(f).name.startswith("build.gradle") for f in repo_map.files):
            return "Gradle"
        return None

    def _detect_go_framework(self, file_contents: dict[str, str]) -> str | None:
        text = "\n".join(file_contents.values()).lower()
        checks = (
            ("github.com/gin-gonic/gin", "Gin"),
            ("github.com/gofiber/fiber", "Fiber"),
            ("github.com/labstack/echo", "Echo"),
        )
        for needle, label in checks:
            if needle in text:
                return label
        return None

    def _detect_rust_framework(self, file_contents: dict[str, str]) -> str | None:
        text = "\n".join(file_contents.values()).lower()
        checks = (
            ("actix-web", "Actix Web"),
            ("axum", "Axum"),
            ("rocket", "Rocket"),
        )
        for needle, label in checks:
            if needle in text:
                return label
        return None
