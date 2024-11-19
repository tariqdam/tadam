def cantor(x: int, y: int) -> int:
    """
    Cantor pairing function. Maps two integers to a single integer.
    :param x: integer
    :param y: integer
    :return: integer
    """
    return (x + y) * (x + y + 1) // 2 + y
