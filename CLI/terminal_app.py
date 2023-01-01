from typing import List

from loguru import logger
import curses.ascii as curscii
from helper_windows import *

logger.add("terminal_app_{time:YYY-MM-DD}.log",
           rotation="1 MB",
           format="{time:HH:mm:ss} | {level} | {message}",
           level="DEBUG",
           backtrace=True,
           diagnose=True)


def display_login_options(window: curses.window, login_options: List[str], option_highlighted: int) -> None:
    window.clear()
    curses.curs_set(0)
    window.border(0)
    curses.noecho()
    login_prompt = "Use arrow keys to go up and down, Press enter to select a choice"
    addstr_center_in_row(window, 0, login_prompt)
    for i, option in enumerate(login_options):
        addstr_center_in_row(window, i + 1, option, curses.A_REVERSE if i == option_highlighted else curses.A_NORMAL)


def login_screen(window: curses.window, login_options: List[str]) -> int:
    option_highlighted = 0
    while True:
        display_login_options(window, login_options, option_highlighted)
        key = window.getch()
        if key == curses.KEY_UP:
            option_highlighted = (option_highlighted - 1) % len(login_options)
        elif key == curses.KEY_DOWN:
            option_highlighted = (option_highlighted + 1) % len(login_options)
        elif key == curscii.CR or key == curscii.LF:
            return option_highlighted


@logger.catch
def main(std_scr: curses.window):
    std_scr.clear()
    std_scr.resize(30, 100)
    curses.curs_set(1)
    curses.cbreak()
    curses.echo()

    login_options = ["Login", "Continue as guest", "Exit"]

    login_window = create_window_in_center(std_scr, 5, 80)
    login_window.keypad(True)
    res = login_screen(login_window, login_options)
    std_scr.clear()

    if res == 0:  # login
        # TODO: login
        ...
    elif res == 1:  # continue as guest
        # TODO: continue as guest
        ...
    else:
        std_scr.clear()
        std_scr.addstr(0, 0, "Press any key to exit...")
        std_scr.getch()


if __name__ == "__main__":
    curses.wrapper(main)
