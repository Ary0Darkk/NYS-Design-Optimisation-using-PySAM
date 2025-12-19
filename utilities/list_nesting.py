# helper func to modify first element of list or tuple
def replace_1st_order(data, new_val):
    """Replaces data[0] with new_val"""
    if not data:
        return data

    # We take the new value and concatenate it with everything from index 1 onwards
    if isinstance(data, list):
        return [new_val] + data[1:]
    else:
        # For tuples, we must use a comma to ensure (new_val,) is treated as a tuple
        return (new_val,) + data[1:]


# Example:
# [1, 2, 3] -> ["NEW", 2, 3]
# (1, 2, 3) -> ("NEW", 2, 3)


def replace_2nd_order(data, new_val):
    """Replaces data[0][0] with new_val"""
    if not data or not data[0]:
        return data

    # 1. Access the inner container and use our 1st order logic on it
    inner_modified = replace_1st_order(data[0], new_val)

    # 2. Reconstruct the outer container with the modified inner one
    if isinstance(data, list):
        return [inner_modified] + data[1:]
    else:
        return (inner_modified,) + data[1:]


# Example:
# [[1, 2], 3] -> [["NEW", 2], 3]
# ((1, 2), 3) -> (("NEW", 2), 3)
