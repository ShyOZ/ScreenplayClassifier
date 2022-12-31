import curses

from curses.textpad import Textbox, rectangle


def create_textbox(std_scr, y: int, x: int, height: int, width: int) -> Textbox:
    win = curses.newwin(height, width, y, x)
    box = Textbox(win)
    rectangle(std_scr, y - 1, x - 1, y + height, x + width)
    std_scr.refresh()
    return box
