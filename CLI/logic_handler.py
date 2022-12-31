from functools import cache
import more_itertools
from data_paths import USERS_PATH, REPORTS_PATH
from report_dataclasses import User, Report


@cache
def get_users():
    return User.schema().loads(USERS_PATH.read_text(), many=True)


@cache
def get_reports():
    return Report.schema().loads(REPORTS_PATH.read_text(), many=True)


@cache
def sign_in(username: str, password: str) -> User | None:
    search_user = User(username, password)
    return more_itertools.first_true(get_users(), pred=lambda user: user == search_user)


if __name__ == "__main__":
    print(get_users())
    print(get_reports())
    print(sign_in("RanYunger", "RY12696"))
