# build_all_code.py

import os
from datetime import datetime

# ===== CONFIG =====
ROOT_DIR = "."
OUTPUT_FILE = "ALL_CODE.md"

EXCLUDE_DIRS = {
    "venv", "__pycache__", ".git", ".idea", ".vscode",
    "node_modules", "build", "dist", "logs", "data"
}

INCLUDE_EXTENSIONS = (
    ".py", ".md", ".txt", ".json", ".yaml", ".yml", ".ini"
)

IMPORTANT_FILES = {
    "requirements.txt",
    "README.md",
    "run.txt"
}

# ===== HELPERS =====
def is_valid_file(filename):
    return filename.endswith(INCLUDE_EXTENSIONS) or filename in IMPORTANT_FILES


def build_tree(root):
    tree = []
    for root_dir, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

        level = root_dir.replace(root, "").count(os.sep)
        indent = "│   " * level + "├── "
        tree.append(f"{indent}{os.path.basename(root_dir) or '.'}")

        subindent = "│   " * (level + 1)
        for f in files:
            if is_valid_file(f):
                tree.append(f"{subindent}├── {f}")

    return "\n".join(tree)


def read_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except:
        try:
            with open(path, "r", encoding="latin-1") as f:
                return f.read()
        except:
            return f"[ERROR READING FILE: {path}]"


# ===== MAIN =====
def main():
    print("🚀 Script started...")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:

        # HEADER
        out.write("# NLACP Attribute Extractor - FULL CODEBASE\n\n")
        out.write(f"Generated at: {datetime.now()}\n\n")

        # TREE
        out.write("## 📁 Project Structure\n\n```\n")
        out.write(build_tree(ROOT_DIR))
        out.write("\n```\n\n")

        # CODE
        out.write("## 📦 Combined Code\n\n")

        file_count = 0

        for root_dir, dirs, files in os.walk(ROOT_DIR):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

            for file in files:
                if not is_valid_file(file):
                    continue

                path = os.path.join(root_dir, file)

                # skip itself
                if OUTPUT_FILE in path:
                    continue

                file_count += 1

                out.write("\n" + "=" * 80 + "\n")
                out.write(f"FILE: {path}\n")
                out.write("=" * 80 + "\n\n")

                content = read_file(path)

                if file.endswith(".py"):
                    out.write("```python\n")
                else:
                    out.write("```\n")

                out.write(content)
                out.write("\n```\n\n")

        # SUMMARY
        out.write("\n## 📊 Summary\n")
        out.write(f"- Total files included: {file_count}\n")
        out.write("- Excluded folders: " + ", ".join(EXCLUDE_DIRS) + "\n")

    print(f"✅ Generated {OUTPUT_FILE} successfully!")


if __name__ == "__main__":
    main()