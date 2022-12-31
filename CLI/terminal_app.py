import curses
from logic_handler import get_users, get_reports, sign_in
from loguru import logger

logger.add("terminal_app_{time:YYY-MM-DD}.log",
           rotation="1 MB",
           format="{time:HH:mm:ss} | {level} | {message}",
           level="DEBUG",
           backtrace=True,
           diagnose=True)


@logger.catch
def main(std_scr):
    std_scr.clear()
    curses.curs_set(1)
    curses.echo()
    std_scr.addstr(2, 2, "Username:")
    std_scr.addstr(3, 2, "Password:")

    username = std_scr.getstr(2, 12).decode()
    password = std_scr.getstr(3, 12).decode()

    user = sign_in(username, password)

    logger.debug(f'{username=}, {password=}, {user=}')

    if user:
        std_scr.addstr(5, 2, f"Welcome {user.username}!")
    else:
        std_scr.addstr(5, 2, "Invalid username or password!")

    std_scr.refresh()
    std_scr.getkey()


if __name__ == "__main__":
    curses.wrapper(main)
