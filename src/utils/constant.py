import numpy as np

unary_ops = ["0", "1", "{}", "({})**2", "({})**3", "({})**4", "exp({})", "sin({})", "cos({})"]


binary_ops = ["({})+({})", "({})-({})", "({})*({})"]


CONTROLLER_INPUT_SIZE = 20
CONTROLLER_HIDDEN_SIZE = 30

POOL_LIMIT = 20