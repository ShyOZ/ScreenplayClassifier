from enum import Enum
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json, LetterCase, config


class Role(Enum):
    Guest = 0
    User = 1
    Admin = 2


@dataclass_json(letter_case=LetterCase.PASCAL)
@dataclass(eq=True)
class User:
    username: str
    password: str
    role: Role = field(compare=False, default=Role)


@dataclass_json(letter_case=LetterCase.PASCAL)
@dataclass
class Screenplay:
    ID: int = field(metadata=config(field_name="ID"))
    title: str
    actual_genre: str
    actual_subgenre1: str = field(metadata=config(field_name="ActualSubGenre1"))
    actual_subgenre2: str = field(metadata=config(field_name="ActualSubGenre2"))
    genre_percentages: dict[str, float] = field(default_factory=dict, init=False)


@dataclass_json(letter_case=LetterCase.PASCAL)
@dataclass
class Report:
    owner: User = field(default_factory=User)
    screenplay: Screenplay = field(default_factory=Screenplay)


if __name__ == "__main__":
    user1 = User("RanYunger", "RY12696", Role.Admin)
    user2 = User("RanYunger", "RY12696", Role.User)

    print(user1 == user2)