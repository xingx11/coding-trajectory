"""ctpipe - Coding Trajectory automation pipeline."""


def strip_claude_wrapper(raw: str) -> str:
    """Strip Claude output wrappers: [__claude_meta:...] prefixes, heading lines, markdown fences."""
    cleaned = raw.strip()
    while cleaned.startswith("[__claude_meta:"):
        end = cleaned.find("\n")
        if end == -1:
            break
        cleaned = cleaned[end + 1:].strip()
    while cleaned.startswith("# "):
        end = cleaned.find("\n")
        if end == -1:
            break
        cleaned = cleaned[end + 1:].strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines)
    return cleaned
