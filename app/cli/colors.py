"""ANSI color codes and display utilities."""

import os
from datetime import datetime


class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"
    BG_DARK = "\033[48;5;236m"


def clear_screen():
    os.system("clear" if os.name != "nt" else "cls")


def fmt_time(ts):
    if ts is None:
        return "—"
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")


def fmt_uptime(seconds):
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    if h > 0:
        return "{}h {}m {}s".format(h, m, s)
    elif m > 0:
        return "{}m {}s".format(m, s)
    return "{}s".format(s)
