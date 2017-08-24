import numpy as np

import ocr

########################################################################################################################
# MARGINALS CLASS
########################################################################################################################
marg = ocr.Marginals(word_length=5)
print("With uniform values.")
print("densities ?", marg.is_densities())
print("consistent?", marg.is_consistent())
print()
print("With random values.")
marg.unary = np.random.rand(5, ocr.ALPHABET_SIZE)
print("densities ?", marg.is_densities())
print("consistent?", marg.is_consistent())
marg.binary = np.random.rand(4, ocr.ALPHABET_SIZE, ocr.ALPHABET_SIZE)
print("consistent?", marg.is_consistent())
