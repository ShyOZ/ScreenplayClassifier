import curses
import curses.ascii
from loguru import logger
from functools import partial
# import argparse

from helper_windows import create_textbox

SCREEN_SIZE = (100, 30)
logger.add("terminal_app_{time:YYY-MM-DD}.log",
           rotation="1 MB",
           format="{time:HH:mm:ss} | {level} | {message}",
           level="DEBUG",
           backtrace=True,
           diagnose=True)


def ignore_control_characters(c: int) -> int:
    if curses.ascii.isprint(c) or c == curses.KEY_LEFT or c == curses.KEY_RIGHT:
        return c

    if c in (curses.ascii.LF, curses.ascii.CR):
        return curses.ascii.BEL

    return curses.ascii.NUL


@logger.catch
def main(std_scr):
    curses.curs_set(1)
    curses.init_pair(1, curses.COLOR_RED, curses.COLOR_BLACK)
    std_scr.clear()
    std_scr.addstr(0, 0, "Hello World!")
    box = create_textbox(std_scr, 5, 5, 10, 10)
    box.edit(ignore_control_characters)
    std_scr.refresh()
    std_scr.getkey()

    curses.endwin()


if __name__ == "__main__":
    curses.wrapper(main)
    # arg_parser = argparse.ArgumentParser()
    # arg_parser.add_argument("-u", "--users", help="users' JSON file path", required=True)
    # arg_parser.add_argument("-r", "--reports", help="reports' JSON file path", required=True)
    #
    # args = arg_parser.parse_args()
    # print(f'{args=}')
