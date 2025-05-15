import numpy as np

unary_ops = ["0", "1", "{}", "({})**2", "({})**3", "({})**4", "exp({})", "sin({})", "cos({})"]


binary_ops = ["({})+({})", "({})-({})", "({})*({})"]


PMF_SIZES = [9, 3, 9, 9, 9, 3, 9,9, 9,3, 9, 9]
CONTROLLER_INPUT_SIZE = 20
CONTROLLER_HIDDEN_SIZE = 30