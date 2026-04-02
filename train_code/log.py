import logging
import os
import re
import sys


# ---- Auto-detect color support ----
# Colors are enabled when stdout is a real terminal (not piped/redirected).
# Override with environment variable: NO_COLOR=1 to force disable.
_COLOR_ENABLED = (
    hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
    and os.environ.get('NO_COLOR') is None
)

_ANSI_RE = re.compile(r'\033\[[0-9;]*m')


class Color:
    Black = 0
    Red = 1
    Green = 2
    Yellow = 3
    Blue = 4
    Magenta = 5
    Cyan = 6
    White = 7


class Mode:
    Foreground = 30
    Background = 40
    ForegroundBright = 90
    BackgroundBright = 100


def tcolor(txt, c, m=Mode.Foreground):
    """Apply ANSI color escape to text. Falls back to plain text if unsupported."""
    if _COLOR_ENABLED:
        return '\033[{}m'.format(m + c) + txt + '\033[0m'
    return txt


def gradient_num_color(value, v_min=37., v_max=42., ascending=True):
    """Apply 24-bit gradient color to a numeric value. Falls back to plain text if unsupported."""
    if _COLOR_ENABLED:
        c_value = min(v_max, (max(v_min, value)))
        color = (c_value - v_min) / (v_max - v_min) * 255
        if ascending:
            color = 255 - color
        r = int(color)
        g = int(255 - color)
        return '\033[38;2;{};{};{}m'.format(r, g, 0) + "{:.2f}".format(value) + '\033[0m'
    return "{:.2f}".format(value)


class _StripAnsiFormatter(logging.Formatter):
    """Formatter that strips ANSI escape codes for clean log files."""

    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)

    def format(self, record):
        msg = super().format(record)
        return _ANSI_RE.sub('', msg)


def gen_log(model_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fmt = "%(asctime)s - %(levelname)s: %(message)s"

    # File handler — strip ANSI codes for clean log files
    log_file = model_path + '/log.txt'
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(_StripAnsiFormatter(fmt))

    # Console handler — keep ANSI codes for terminal display
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(fmt))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
