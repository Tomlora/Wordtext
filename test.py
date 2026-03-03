import re
import numpy as np

def decode_unicode_escapes_batch(s: pl.Series) -> pl.Series:
    pattern = re.compile(r'\\u([0-9a-fA-F]{4})')
    vectorized = np.vectorize(
        lambda x: pattern.sub(lambda m: chr(int(m.group(1), 16)), x) if isinstance(x, str) else x
    )
    return pl.Series(vectorized(s.to_numpy(allow_copy=True)))

casts = [
    pl.col(c).cast(pl.Utf8, strict=False).map_batches(
        decode_unicode_escapes_batch,
        return_dtype=pl.Utf8
    )
    for c in cols
]
