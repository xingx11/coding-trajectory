"""GitHub project search and clone utilities."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

try:
    import requests as _requests
except ImportError:
    _requests = None


@dataclass
class GitHubRepo:
    full_name: str
    clone_url: str
    description: str
    language: str
    stars: int
    updated_at: str


DOMAIN_SEARCH_TERMS: dict[str, str] = {
    "web_frontend": "frontend OR react OR vue OR angular OR nextjs OR svelte",
    "backend_service": "api OR server OR backend OR rest OR microservice OR web-framework",
    "mobile_app": "android OR ios OR react-native OR flutter OR mobile-app",
    "data_engineering": "etl OR pipeline OR data-pipeline OR spark OR airflow OR streaming",
    "ai_ml": "machine-learning OR deep-learning OR model-training OR inference OR mlops",
    "devtools_test": "linter OR bundler OR test-framework OR cli-tool OR developer-tools",
    "devops_infrastructure": "docker OR kubernetes OR terraform OR ci-cd OR monitoring OR devops",
    "game_dev": "game OR game-engine OR unity OR godot OR gamedev",
    "database_storage": "database OR orm OR cache OR migration OR storage",
    "desktop_gui": "electron OR tauri OR qt OR desktop-app OR gui",
    "graphics_media": "rendering OR visualization OR image-processing OR video OR graphics",
    "business_logic": "order OR inventory OR pricing OR workflow OR erp OR crm",
    "docs_knowledge": "documentation OR wiki OR knowledge-base OR static-site OR docs",
    "embedded_system": "firmware OR rtos OR iot OR embedded OR driver OR hardware",
    "pkg_manager_cli": "package-manager OR cli OR command-line OR installer",
    "operating_system": "kernel OR filesystem OR scheduler OR operating-system",
    "security_auth": "security OR authentication OR authorization OR cryptography",
    "lang_runtime": "compiler OR interpreter OR runtime OR virtual-machine OR language",
    "scientific_computing": "simulation OR numerical OR hpc OR scientific-computing",
    "cms_ecommerce": "cms OR ecommerce OR storefront OR content-management",
    "blockchain_web3": "blockchain OR smart-contract OR web3 OR ethereum",
}

LANGUAGE_MAP: dict[str, str] = {
    "ts": "TypeScript",
    "js": "JavaScript",
    "python": "Python",
    "java": "Java",
    "go": "Go",
    "c": "C",
    "c++": "C++",
    "rust": "Rust",
    "kotlin": "Kotlin",
    "swift": "Swift",
    "dart": "Dart",
    "lua": "Lua",
    "shell": "Shell",
    "html/css": "HTML",
    "sql": "PLpgSQL",
    "ruby": "Ruby",
    "c#": "C#",
    "php": "PHP",
    "scala": "Scala",
    "r": "R",
    "other": "",
}


def _resolve_proxy(http_proxy: str = "") -> str:
    """Resolve proxy from explicit arg or environment variables."""
    if http_proxy:
        return http_proxy
    return os.environ.get("HTTPS_PROXY", "") or os.environ.get("HTTP_PROXY", "") or os.environ.get("https_proxy", "") or os.environ.get("http_proxy", "")



def _fetch_via_curl(
    url: str,
    headers: dict[str, str],
    proxy: str = "",
    timeout: int = 30,
) -> bytes | None:
    """Fallback: fetch via system curl (better proxy/TLS handling)."""
    curl = shutil.which("curl")
    if not curl:
        return None
    cmd = [curl, "-s", "-f", "--max-time", str(timeout)]
    for k, v in headers.items():
        cmd += ["-H", f"{k}: {v}"]
    if proxy:
        cmd += ["--proxy", proxy]
    cmd.append(url)
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=timeout + 10)
        if result.returncode == 0:
            return result.stdout
    except (subprocess.TimeoutExpired, OSError):
        pass
    return None


def _fetch_json(
    url: str,
    headers: dict[str, str],
    http_proxy: str = "",
    timeout: int = 30,
    max_retries: int = 3,
) -> dict | None:
    """Fetch JSON from a URL with retries. Prefers requests, falls back to curl."""
    proxy = _resolve_proxy(http_proxy)

    if _requests is not None:
        proxies = {"http": proxy, "https": proxy} if proxy else None
        for attempt in range(max_retries):
            if attempt > 0:
                wait = min(2 ** attempt, 10)
                print(f"  [retry {attempt}/{max_retries - 1}] waiting {wait}s...")
                time.sleep(wait)
            try:
                resp = _requests.get(url, headers=headers, proxies=proxies, timeout=timeout)
                if resp.status_code == 403:
                    retry_after = int(resp.headers.get("Retry-After", "60"))
                    print(f"  WARNING: GitHub rate limit, waiting {retry_after}s...")
                    time.sleep(retry_after)
                    continue
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                last_error = str(exc)
        print(f"  WARNING: requests failed after {max_retries} attempts ({last_error}), trying curl...")
    else:
        print(f"  WARNING: requests library not installed, trying curl...")

    raw = _fetch_via_curl(url, headers, proxy, timeout)
    if raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            print(f"  WARNING: curl returned invalid JSON")

    print(f"  ERROR: all fetch methods failed for {url[:80]}")
    return None


def search_projects(
    domain: str,
    language: str,
    count: int = 5,
    min_stars: int = 50,
    exclude_repos: set[str] | None = None,
    github_token: str = "",
    http_proxy: str = "",
) -> list[GitHubRepo]:
    """Search GitHub for projects matching domain and language criteria."""
    exclude = exclude_repos or set()
    topic_terms = DOMAIN_SEARCH_TERMS.get(domain, domain.replace("_", " "))
    gh_lang = LANGUAGE_MAP.get(language, language)

    query_parts = [topic_terms]
    if gh_lang:
        query_parts.append(f"language:{gh_lang}")
    query_parts.append(f"stars:{min_stars}..5000")
    query_parts.append("fork:false")
    query_parts.append("archived:false")
    query_parts.append("size:<100000")
    query = " ".join(query_parts)

    params = urllib.parse.urlencode({
        "q": query,
        "sort": "stars",
        "order": "desc",
        "per_page": min(count * 3, 30),
    })
    url = f"https://api.github.com/search/repositories?{params}"

    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "ctpipe/1.0",
    }
    if github_token:
        headers["Authorization"] = f"token {github_token}"

    print(f"  [github] Querying: {query[:80]}...")
    t0 = time.time()
    data = _fetch_json(url, headers, http_proxy=http_proxy)
    print(f"  [github] Search API returned in {time.time() - t0:.1f}s")
    if data is None:
        return []

    repos: list[GitHubRepo] = []
    for item in data.get("items", []):
        full_name = item.get("full_name", "")
        if full_name in exclude:
            continue
        repos.append(GitHubRepo(
            full_name=full_name,
            clone_url=item.get("clone_url", ""),
            description=(item.get("description") or "")[:200],
            language=item.get("language") or "",
            stars=item.get("stargazers_count", 0),
            updated_at=item.get("updated_at", ""),
        ))
        if len(repos) >= count:
            break

    print(f"  [github] Found {len(repos)} candidate repos")
    return repos


def clone_project(
    repo: GitHubRepo,
    dest_root: Path,
    task_id: str,
    http_proxy: str = "",
) -> Path | None:
    """Shallow clone a GitHub repo into dest_root/task_id/repo_name."""
    repo_name = repo.full_name.split("/")[-1] if "/" in repo.full_name else repo.full_name
    dest = dest_root / task_id / repo_name
    if dest.exists():
        print(f"  [skip] {dest} already exists")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    proxy = _resolve_proxy(http_proxy)
    cmd = ["git"]
    if proxy:
        cmd += ["-c", f"http.proxy={proxy}", "-c", f"https.proxy={proxy}"]
    cmd += ["clone", "--depth", "1", "--filter=blob:none", repo.clone_url, str(dest)]
    print(f"  [git] Cloning {repo.full_name}...")
    t0 = time.time()
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=90)
        print(f"  [git] Cloned in {time.time() - t0:.1f}s -> {dest}")
        return dest
    except subprocess.TimeoutExpired:
        print(f"  WARNING: clone timed out after {time.time() - t0:.0f}s for {repo.full_name}")
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        return None
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace")[:200] if exc.stderr else ""
        print(f"  WARNING: clone failed for {repo.full_name}: {stderr}")
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        return None


def search_and_clone(
    domain: str,
    language: str,
    task_id: str,
    dest_root: Path,
    exclude_repos: set[str] | None = None,
    github_token: str = "",
    http_proxy: str = "",
) -> tuple[GitHubRepo, Path] | None:
    """Search for a project and clone the first successful one."""
    repos = search_projects(domain, language, count=5, exclude_repos=exclude_repos, github_token=github_token, http_proxy=http_proxy)
    if not repos:
        print(f"  WARNING: no repos found for {domain}/{language}")
        return None

    for repo in repos:
        path = clone_project(repo, dest_root, task_id, http_proxy=http_proxy)
        if path:
            return repo, path
        time.sleep(1)

    print(f"  WARNING: all clone attempts failed for {domain}/{language}")
    return None
