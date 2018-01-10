from chunk.features import Features
from chunk.parse import build_dictionary, read_data

filename = "../../data/conll2000/train.att.txt"
dattributes = build_dictionary(filename)
x, y = read_data(filename, dattributes)

print(len(dattributes.keys()))
print(y[:10])
print(x[0])

feat = Features()
feat.add_dictionary(x, y)
feat.display()

# probas, log_part = feat.infer_probabilities(x[0])
# probas.display(Features.TAGS)
#
# foo = Features()
# uniform = uniform(len(x[0]), nb_tags, log=False)
# foo.add_centroid(x[0], uniform)
# foo.add_centroid(x[0], probas)
# foo.display()
#
# ypred, _ = feat.predict(x[0])
# print("reference:", [Features.TAGS[tag] for tag in y[0]])
# print("prediction:", [Features.TAGS[tag] for tag in ypred])
# print()
# print(feat.word_score(x[0], y[0]))
#
# feat = feat.multiply_scalar(2)
# feat.add(foo)
# print(feat.squared_norm())
# print(feat.reduce())
# print(feat.inner_product(foo))
