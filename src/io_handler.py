import os

def safe_write(filepath, content, encoding='utf-8'):
    # We use 'wb' and encode manually to ensure NO OS-level newline translation
    temp_path = f"{filepath}.tmp"
    try:
        with open(temp_path, 'wb') as f:
            f.write(content.encode(encoding))
        os.replace(temp_path, filepath)
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e
