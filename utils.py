import numpy as np

OCR_IMAGE_HEIGHT = 16
OCR_IMAGE_WIDTH = 8

def entropy(logproba, axis=None, returnlog=False):
    themax = np.amax(logproba)
    try:
        ans = themax + np.log(- np.sum(np.exp(logproba - themax) * logproba, axis=axis))
        if returnlog:
            return ans
        else:
            return np.exp(ans)
    except FloatingPointError:
        print("Entropy problem:",
              themax, "\n",
              logproba)
        raise


def kullback_leibler(logp, logq, axis=None, returnlog=False):
    themax = np.amax(logp)
    ans = themax + np.log(np.sum(np.exp(logp - themax) * (logp - logq), axis=axis))
    if returnlog:
        return ans
    else:
        try:
            return np.exp(ans)
        except FloatingPointError:
            print("too big ", ans)
            raise

def boolean_encoding(y, k):
    """Return the n*k matrix Y whose line i is the one-hot encoding of y_i."""
    n = y.shape[0]
    ans = np.zeros([n, k])
    ans[np.arange(n), y] = 1
    return ans


def letters2wordimage(letters_images):
    word_image = np.zeros([OCR_IMAGE_HEIGHT, 2])
    spacing = np.zeros([OCR_IMAGE_HEIGHT, 2])
    for letter in letters_images:
        letter_image = letter.reshape((OCR_IMAGE_HEIGHT, OCR_IMAGE_WIDTH))
        word_image = np.hstack((word_image, letter_image, spacing))
    return word_image
