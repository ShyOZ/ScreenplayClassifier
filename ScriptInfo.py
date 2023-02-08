from dataclasses import dataclass, field
from typing import Set

from dataclasses_json import dataclass_json, config
from marshmallow import fields


@dataclass_json
@dataclass(frozen=True)
class ScriptInfo:
    title: str
    info_url: str
    script_url: str
    genres: Set[str] = field(metadata=config(encoder=list,
                                             decoder=set,
                                             mm_field=fields.List(fields.String())),
                             default=set)
