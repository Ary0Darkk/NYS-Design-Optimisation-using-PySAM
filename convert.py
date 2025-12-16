import matlab


def convert(value):
    if isinstance(value, (int, float, str, bool, type(None))):
        return value

    if isinstance(value, (list, tuple)):
        return [convert(v) for v in value]

    if isinstance(value, matlab.double):
        rows, cols = value.size

        # scalar case with bug fix
        if (rows, cols) == (1, 1):
            data = value._data[0]

            if isinstance(data, list):  # MATLAB bug case
                try:
                    return [float(x) for x in data]  # vector
                except:
                    return [convert(x) for x in data]  # nested matrix

            return float(data)

        # vector
        if rows == 1 or cols == 1:
            return [float(x) for x in value._data]

        # matrix
        return [[float(value[r][c]) for c in range(cols)] for r in range(rows)]

    if isinstance(value, matlab.logical):
        return bool(value[0][0])

    if isinstance(value, (matlab.int32, matlab.uint8)):
        return int(value._data[0])

    if hasattr(value, "_fieldnames"):
        return {f: convert(getattr(value, f)) for f in value._fieldnames}

    return value
