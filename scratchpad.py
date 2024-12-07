from simulib.config.config import kwargs


print(kwargs)


def a(hey, **kwargs):
    print(hey)


b = {"hey": 2, "five": 3}

a(**b)