import numpy as np


def entropy(logproba, returnlog=False):
    themax = np.amax(logproba)
    try:
        ans = themax + np.log(- np.sum(np.exp(logproba - themax) * logproba))
        if returnlog:
            return ans
        else:
            return np.exp(ans)
    except FloatingPointError:
        print("Entropy problem:",
              themax, "\n",
              logproba)
        raise


def kullback_leibler(logp, logq, returnlog=False):
    themax = np.amax(logp)
    tmp = np.sum(np.exp(logp - themax) * (logp - logq))
    if tmp <= -1e-8:
        raise Warning(f"Numerical stability: {tmp}")
    if tmp <= 0:
        if returnlog:
            return -np.infty
        else:
            return 0

    ans = themax + np.log(tmp)
    if returnlog:
        return ans
    else:
        try:
            return np.exp(ans)
        except FloatingPointError:
            print("too big ", ans)
            raise


def logsumexp(arr, axis=None):
    themax = np.amax(arr)
    return themax + np.log(np.sum(np.exp(arr - themax), axis=axis))


def logsubtractexp(x1, x2):
    themax = max(np.amax(x1), np.amax(x2))
    expvalue = np.exp(x1 - themax) - np.exp(x2 - themax)
    sign = np.sign(expvalue)
    ans = themax + np.log(np.absolute(expvalue))
    return ans, sign


def letters2wordimage(letters_images):
    OCR_IMAGE_HEIGHT = 16
    OCR_IMAGE_WIDTH = 8
    word_image = np.zeros([OCR_IMAGE_HEIGHT, 2])
    spacing = np.zeros([OCR_IMAGE_HEIGHT, 2])
    for letter in letters_images:
        letter_image = letter.reshape((OCR_IMAGE_HEIGHT, OCR_IMAGE_WIDTH))
        word_image = np.hstack((word_image, letter_image, spacing))
    return word_image
