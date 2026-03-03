import re

pattern = re.compile(r'\\u([0-9a-fA-F]{4})')

def decode_unicode_escapes(s):
    if not isinstance(s, str):
        return s
    return pattern.sub(lambda m: chr(int(m.group(1), 16)), s)

casts = [
    pl.col(c).cast(pl.Utf8, strict=False).map_elements(
        decode_unicode_escapes,
        return_dtype=pl.Utf8
    )
    for c in cols
]
