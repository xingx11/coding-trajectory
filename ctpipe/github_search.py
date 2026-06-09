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

from ctpipe.config import is_safe_clone_url

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
    """Fallback: fetch via system curl (better proxy/TLS handling).

    Headers are passed via --config stdin to avoid exposing tokens in
    the process command line.
    """
    curl = shutil.which("curl")
    if not curl:
        return None
    cmd = [curl, "-s", "-f", "--max-time", str(timeout), "--config", "-"]
    if proxy:
        cmd += ["--proxy", proxy]

    # Build curl config: headers + URL via stdin (not command-line args)
    config_lines: list[str] = []
    for k, v in headers.items():
        config_lines.append(f'header = "{k}: {v}"')
    config_lines.append(f'url = "{url}"')
    config_input = "\n".join(config_lines).encode("utf-8")

    try:
        result = subprocess.run(cmd, input=config_input, capture_output=True, timeout=timeout + 10)
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

    last_error = ""
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
                    retry_after = min(int(resp.headers.get("Retry-After", "60")), 120)
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
    base_url = f"https://api.github.com/search/repositories?{params}"

    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "ctpipe/1.0",
    }
    if github_token:
        headers["Authorization"] = f"token {github_token}"

    repos: list[GitHubRepo] = []
    max_pages = max(3, 1 + len(exclude) // 30)

    for page in range(1, max_pages + 1):
        url = f"{base_url}&page={page}"
        print(f"  [github] Querying page {page}: {query[:80]}...")
        t0 = time.time()
        data = _fetch_json(url, headers, http_proxy=http_proxy)
        print(f"  [github] Search API returned in {time.time() - t0:.1f}s")
        if data is None:
            break

        items = data.get("items", [])
        if not items:
            break

        for item in items:
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
    # Validate clone URL to prevent command injection or unsafe protocols
    if not is_safe_clone_url(repo.clone_url):
        print(f"  WARNING: unsafe clone URL rejected: {repo.clone_url[:80]}")
        return None
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


# ---------------------------------------------------------------------------
# Gitee support
# ---------------------------------------------------------------------------

_GITEE_LANG_OVERRIDES: dict[str, str] = {"sql": ""}

GITEE_DOMAIN_SEARCH_TERMS: dict[str, str] = {
    "web_frontend": "frontend vue react",
    "backend_service": "api server backend",
    "mobile_app": "android ios mobile",
    "data_engineering": "etl pipeline data",
    "ai_ml": "machine-learning deep-learning",
    "devtools_test": "test framework cli",
    "devops_infrastructure": "docker kubernetes devops",
    "game_dev": "game 游戏",
    "database_storage": "database orm cache",
    "desktop_gui": "electron desktop gui",
    "graphics_media": "rendering visualization image",
    "business_logic": "order workflow erp",
    "docs_knowledge": "documentation wiki docs",
    "embedded_system": "firmware iot embedded",
    "pkg_manager_cli": "cli 命令行",
    "operating_system": "kernel filesystem operating-system",
    "security_auth": "security authentication authorization",
    "lang_runtime": "compiler interpreter runtime",
    "scientific_computing": "simulation numerical scientific",
    "cms_ecommerce": "cms ecommerce storefront",
    "blockchain_web3": "blockchain smart-contract web3",
}


_GITEE_WIDGET_ID = "wong1slagnlmzwvsu5ya"
_GITEE_SEARCH_URL = f"https://so.gitee.com/v1/search/widget/{_GITEE_WIDGET_ID}"


def _gitee_search_page(
    query: str,
    headers: dict[str, str],
    page_size: int,
    max_pages: int,
    min_stars: int,
    exclude: set[str],
    count: int,
) -> list[GitHubRepo]:
    """Run paginated Indexea widget search and return filtered repos."""
    repos: list[GitHubRepo] = []
    for page in range(max_pages):
        qs = urllib.parse.urlencode({"q": query, "size": page_size, "from": page * page_size})
        url = f"{_GITEE_SEARCH_URL}?{qs}"
        print(f"  [gitee] Querying page {page + 1}: {query[:60]}...")
        t0 = time.time()
        data = _fetch_json(url, headers, http_proxy="")
        print(f"  [gitee] Search API returned in {time.time() - t0:.1f}s")
        if data is None:
            break

        hits = (data.get("hits") or {}).get("hits") or []
        if not hits:
            break

        for hit in hits:
            fields = hit.get("fields") or {}
            repo_url = (fields.get("url") or [""])[0]
            if not repo_url:
                continue
            full_name = repo_url.removeprefix("https://gitee.com/").strip("/")
            stars = (fields.get("count.star") or [0])[0]
            langs = fields.get("langs") or []
            is_fork = (fields.get("fork") or [0])[0]

            if full_name in exclude:
                continue
            if is_fork:
                continue
            if stars < min_stars:
                continue

            clone_url = f"{repo_url}.git"
            if not is_safe_clone_url(clone_url):
                continue
            repos.append(GitHubRepo(
                full_name=full_name,
                clone_url=clone_url,
                description=((fields.get("description") or [""])[0])[:200],
                language=langs[0] if langs else "",
                stars=stars,
                updated_at=((fields.get("last_push_at") or [""])[0]),
            ))
            if len(repos) >= count:
                break

        if len(repos) >= count:
            break

    return repos


def search_projects_gitee(
    domain: str,
    language: str,
    count: int = 5,
    min_stars: int = 30,
    exclude_repos: set[str] | None = None,
    gitee_token: str = "",
) -> list[GitHubRepo]:
    """Search Gitee for projects matching domain and language criteria.

    Uses the so.gitee.com Indexea widget search (the v5 search API no longer
    returns public results for personal access tokens).
    """
    exclude = exclude_repos or set()
    raw_terms = GITEE_DOMAIN_SEARCH_TERMS.get(domain, DOMAIN_SEARCH_TERMS.get(domain, domain.replace("_", " ")))
    domain_kw = raw_terms.split()[:2]
    gitee_lang = _GITEE_LANG_OVERRIDES.get(language, LANGUAGE_MAP.get(language, language))

    headers = {
        "Accept": "application/json",
        "User-Agent": "ctpipe/1.0",
    }
    page_size = 50
    max_pages = max(3, 1 + len(exclude) // page_size)

    if gitee_lang:
        query = " ".join(domain_kw + [gitee_lang])
        repos = _gitee_search_page(query, headers, page_size, max_pages, min_stars, exclude, count)
        if len(repos) >= count:
            print(f"  [gitee] Found {len(repos)} candidate repos")
            return repos
        already = {r.full_name for r in repos}
        exclude = exclude | already

    query_fallback = " ".join(domain_kw)
    repos_fb = _gitee_search_page(query_fallback, headers, page_size, max_pages, min_stars, exclude, count - len(repos) if gitee_lang else count)
    if gitee_lang:
        repos.extend(repos_fb)
    else:
        repos = repos_fb

    print(f"  [gitee] Found {len(repos)} candidate repos")
    return repos


def search_and_clone_gitee(
    domain: str,
    language: str,
    task_id: str,
    dest_root: Path,
    exclude_repos: set[str] | None = None,
    gitee_token: str = "",
    http_proxy: str = "",
) -> tuple[GitHubRepo, Path] | None:
    """Search Gitee for a project and clone the first successful one."""
    repos = search_projects_gitee(
        domain, language, count=5,
        exclude_repos=exclude_repos, gitee_token=gitee_token,
    )
    if not repos:
        print(f"  WARNING: no repos found on Gitee for {domain}/{language}")
        return None

    for repo in repos:
        path = clone_project(repo, dest_root, task_id, http_proxy=http_proxy)
        if path:
            return repo, path
        time.sleep(1)

    print(f"  WARNING: all Gitee clone attempts failed for {domain}/{language}")
    return None
