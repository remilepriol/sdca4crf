import numpy as np

import ocr

########################################################################################################################
# MARGINALS CLASS
########################################################################################################################
word = ocr.word2list("beta")
word_length = len(word)
marg = ocr.Probability(word_length=word_length)
print("With uniform values.")
print("densities ?", marg.is_density())
print("consistent?", marg.is_consistent())
print()
print("With random values.")
marg.unary = np.random.rand(word_length, ocr.ALPHABET_SIZE)
print("densities ?", marg.is_density())
print("consistent?", marg.is_consistent())
marg.binary = np.random.rand(word_length - 1, ocr.ALPHABET_SIZE, ocr.ALPHABET_SIZE)
print("consistent?", marg.is_consistent(), "\n")

marg = ocr.LogProbability(word_length=word_length)
