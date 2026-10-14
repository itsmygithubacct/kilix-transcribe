"""Inert schema control validator for the exact published keyword subset.
Not a replacement for a future independent Draft2020-12 qualification validator.
"""
import math
import re
from .fd import require, Refusal
from .numeric_limits import MIN_INTEGER, MAX_INTEGER

def validate(value, schema):
    supported = {'$schema','$id','type','properties','required','additionalProperties',
                 'items','maxItems','minimum','maximum','minLength','maxLength','pattern',
                 'enum','const','anyOf'}
    require(set(schema) <= supported, 'UNSUPPORTED_SCHEMA_KEYWORD')
    if 'anyOf' in schema:
        for option in schema['anyOf']:
            try:
                validate(value,option)
                return
            except Refusal:
                pass
        raise Refusal('SCHEMA_ANYOF_MISMATCH')
    if 'const' in schema:
        require(type(value) is type(schema['const']) and value == schema['const'], 'SCHEMA_CONST_MISMATCH')
    if 'enum' in schema:
        require(any(type(value) is type(x) and value == x for x in schema['enum']), 'SCHEMA_ENUM_MISMATCH')
    kind = schema.get('type')
    if kind:
        types = {'integer':(int,), 'number':(int,float),'string':(str,),'null':(type(None),),
                 'object':(dict,), 'array':(list,), 'boolean':(bool,)}
        require(type(value) in types[kind], 'SCHEMA_TYPE_MISMATCH')
    if type(value) in (float,int):
        if type(value) is float:
            require(math.isfinite(value), 'SCHEMA_NONFINITE')
        if 'minimum' in schema: require(value >= schema['minimum'],'SCHEMA_MINIMUM')
        if 'maximum' in schema: require(value <= schema['maximum'],'SCHEMA_MAXIMUM')
        if type(value) is int:
            require(MIN_INTEGER <= value <= MAX_INTEGER, 'SCHEMA_NUMBER_RANGE')
    if kind == 'string':
        require(schema.get('minLength',0) <= len(value) <= schema.get('maxLength',1<<30),'SCHEMA_STRING_SIZE')
        if 'pattern' in schema: require(re.fullmatch(schema['pattern'],value) is not None,'SCHEMA_PATTERN')
    if kind == 'array':
        require(len(value) <= schema.get('maxItems',1<<30), 'SCHEMA_ARRAY_LIMIT')
        for item in value: validate(item,schema['items'])
    if kind == 'object':
        require(set(schema.get('required',[])) <= set(value),'SCHEMA_REQUIRED')
        if schema.get('additionalProperties') is False:
            require(set(value) <= set(schema['properties']),'SCHEMA_UNKNOWN_FIELD')
        for key,item in value.items(): validate(item,schema['properties'][key])
