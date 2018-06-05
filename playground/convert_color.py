

s = '#337ab7'


def make_lighter(s, factor):
    r = int(int(s[1:3], 16) * factor)
    g = int(int(s[3:5], 16) * factor)
    b = int(int(s[5:], 16) * factor)

    return f"#{hex(r)[2:]}{hex(g)[2:]}{hex(b)[2:]}"


if __name__ == '__main__':
    print(make_lighter(s, 1.39))



