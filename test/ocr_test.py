import numpy as np

import ocr

########################################################################################################################
# MARGINALS CLASS
########################################################################################################################
word = ocr.word2list("beta")
word_length = len(word)
marg = ocr.Probability(word_length=word_length)
print("With uniform values.")
print("densities ?", marg.is_densities())
print("consistent?", marg.is_consistent())
print()
print("With random values.")
marg.unary = np.random.rand(word_length, ocr.ALPHABET_SIZE)
print("densities ?", marg.is_densities())
print("consistent?", marg.is_consistent())
marg.binary = np.random.rand(word_length - 1, ocr.ALPHABET_SIZE, ocr.ALPHABET_SIZE)
print("consistent?", marg.is_consistent(), "\n")

marg = ocr.Probability(word_length=word_length)
u2 = np.zeros([word_length, ocr.ALPHABET_SIZE])
b2 = np.zeros([word_length - 1, ocr.ALPHABET_SIZE, ocr.ALPHABET_SIZE])
u2[np.arange(word_length), word] = 1
b2[np.arange(word_length - 1), word[:-1], word[1:]] = 1
marg2 = ocr.Probability(unary=u2, binary=b2)
print("Uniform entropy :", marg.entropy(), "\t T*log(26) :", word_length * np.log(ocr.ALPHABET_SIZE))
print("Dirac Entropy :", marg2.entropy())
print("Divergence :", marg2.kullback_leibler(marg))
