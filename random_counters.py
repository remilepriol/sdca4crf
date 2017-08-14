import numpy as np


class RandomCounters:
    """Implement the random counters algorithm from 'Efficiency of coordinate descent methods on huge-scale
    optimization problems', by Yuri Nesterov, 2012. It is a binary search over changing scores.
    Complexity of changing one score is O(log n). Complexity of sampling is O(log n). Complexity of initialization is
    O(n log n)."""

    def __init__(self, score_leaves):
        self.score_tree = [score_leaves]
        score_level = self.score_tree[-1]
        n = score_level.shape[0]
        while n > 1:
            if n % 2 == 0:
                upper_level = score_level[::2] + score_level[1::2]
                self.score_tree.append(upper_level)
            else:
                upper_level = np.empty(n // 2 + 1)
                upper_level[:-1] = score_level[:-1:2] + score_level[1::2]
                upper_level[-1] = score_level[-1]
                self.score_tree.append(upper_level)
            score_level = self.score_tree[-1]
            n = score_level.shape[0]

    def update(self, new_score, index):
        lower_level = self.score_tree[0]
        lower_level[index] = new_score
        for level in self.score_tree[1:]:
            if index + 1 == lower_level.shape[0] and (index + 1) % 2 != 0:  # end of line, transfer
                level[-1] = lower_level[index]
            else:
                new_index = index // 2
                level[new_index] = lower_level[2 * new_index] + lower_level[2 * new_index + 1]
            index //= 2
            lower_level = level

    def sample(self):
        score_sum = self.score_tree[-1][0]
        random_value = np.random.rand() * score_sum
        index = 0
        for level in self.score_tree[-2::-1]:
            if (index + 1) * 2 > level.shape[0]:  # end of line, transfer
                index = index * 2
            elif random_value < level[index * 2]:
                index = index * 2
            else:
                random_value -= level[index * 2]
                index = index * 2 + 1
        return index
