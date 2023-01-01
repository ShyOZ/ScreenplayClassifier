import curses
from curses.textpad import Textbox, rectangle


def create_textbox(window: curses.window, y: int, x: int, height: int, width: int) -> Textbox:
    win = window.derwin(height, width, y, x)
    box = Textbox(win)
    rectangle(window, y - 1, x - 1, y + height, x + width)
    window.refresh()
    return box


def create_window_in_center(window: curses.window, height: int, width: int):
    start_y = (curses.LINES - height) // 2
    start_x = (curses.COLS - width) // 2
    win = window.derwin(height, width, start_y, start_x)
    return win


def addstr_center_in_row(window: curses.window, row: int, text: str, attr: int = curses.A_NORMAL):
    col = (window.getmaxyx()[1] - len(text)) // 2
    window.addstr(row, col, text, attr)


def addstr_center_in_column(window: curses.window, col: int, text: str, attr: int = curses.A_NORMAL):
    row = window.getmaxyx()[0] // 2
    window.addstr(row, col, text, attr)
