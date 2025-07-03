import numpy as np

unary_ops = ["0", "1", "{}", "({})**2", "({})**3", "({})**4", "exp({})", "sin({})", "cos({})"]


binary_ops = ["({})+({})", "({})-({})", "({})*({})"]


POOL_LIMIT = 20