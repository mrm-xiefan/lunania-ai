#!/usr/bin/env python35
# -*- coding: utf-8 -*-

import logging
from typing import TypeVar, Dict, List
LOGGER = logging.getLogger(__name__)

S = TypeVar('S', Dict[str, any], List[any])


def to_camel_case_dict_key(ob: S, ret: Dict[str, any]) -> any:
    '''
    dict型のkeyをsnake_caseからcamel_caseに変換する。
    valueについても再帰的に変換する。
    '''
    for k, v in ob.items():
        ret[to_camel_case(k)] = to_camel_case_ob(v)

    return ret


def _to_camel_case_from_list(ob: List[any]) -> List[any]:
    ret = []  # type: List[any]
    for v in ob:
        ret.append(to_camel_case_ob(v))

    return ret


def to_camel_case(snake_case: str):
    l = snake_case.split("_")
    return l[0].lower() + "".join(map(str.capitalize, l[1:]))


def to_camel_case_ob(ob: any) -> any:
    if isinstance(ob, dict):
        return to_camel_case_dict_key(ob, {})
    elif isinstance(ob, list):
        return _to_camel_case_from_list(ob)
    else:
        return ob
