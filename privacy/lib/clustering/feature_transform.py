def transform(f_expo_pos, f_expo_neg, f_dens, method = 'abs'):
    """
    Apply feature transform on photo features scaled by focal exposure

    :param f_expo_pos:
    :param f_expo_neg:
    :param f_dens:
    :return:
        transformed features

    """

    if method == 'abs':
        f_abs = abs(f_expo_pos) + abs(f_expo_neg)
        return [f_abs, f_dens]

    if method == 'origin':
        return [f_expo_pos, f_expo_neg, f_dens]