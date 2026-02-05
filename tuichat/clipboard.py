import shutil
import subprocess
import sys


def clipboard_command():
    """The first available clipboard tool for this platform, or None."""
    if sys.platform == "darwin":
        candidates = [["pbcopy"]]
    elif sys.platform.startswith("win"):
        candidates = [["clip"]]
    else:
        candidates = [["wl-copy"], ["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"]]
    for cmd in candidates:
        if shutil.which(cmd[0]):
            return cmd
    return None


def copy_to_clipboard(text: str, run=subprocess.run) -> bool:
    cmd = clipboard_command()
    if cmd is None:
        return False
    try:
        run(cmd, input=text.encode("utf-8"), check=True)
        return True
    except (OSError, subprocess.SubprocessError):
        return False
