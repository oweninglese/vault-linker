import re

def worker_process_file(content, approved_tags):
    # Protect URLs by finding them first
    url_pattern = re.compile(r'https?://[^\s)\]]+')
    urls = list(url_pattern.finditer(content))

    # Sort tags by length (longest first)
    sorted_tags = sorted(approved_tags, key=len, reverse=True)
    # Combine all tags into one giant 'OR' pattern
    tag_regex = re.compile(rf'(?<!\[\[)\b({"|".join(map(re.escape, sorted_tags))})\b(?!\]\])', re.IGNORECASE)

    def is_in_url(pos):
        return any(u.start() <= pos < u.end() for u in urls)

    def replace(match):
        if is_in_url(match.start()):
            return match.group(0)
        return f"[[{match.group(1)}]]"

    new_text, count = tag_regex.subn(replace, content)
    return new_text, count
