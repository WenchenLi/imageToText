import enchant
from collections import *


class SpellChecker:
    def __init__(self, lan="en_US"):
        self.dict = enchant.Dict(lan)
        self.string = []
        self.bow = collections.defaultdict(int)

    def update_bow(self, string):
        self.string = string.split()
        string = self.string
        for s in string:
            if not self.dict.check(s):
	            for w in self.dict.suggest(s):
	                self.bow[w] += 1

	def update_string(self):
		pass


    def minDistance(self, word1, word2):
        """ 
        add minimum edit distance for two string as the score
                Time:  O(n * m)
                Space: O(n + m)
                Given two words word1 and word2, find the minimum number of steps
                required to convert word1 to word2. (each operation is counted as 1 step.)
                a) Insert a character
                b) Delete a character
                c) Replace a character
        """
        distance = [[i] for i in xrange(len(word1) + 1)]
        distance[0] = [j for j in xrange(len(word2) + 1)]

        for i in xrange(1, len(word1) + 1):
            for j in xrange(1, len(word2) + 1):
                insert = distance[i][j - 1] + 1
                delete = distance[i - 1][j] + 1
                replace = distance[i - 1][j - 1]
                if word1[i - 1] != word2[j - 1]:
                    replace += 1
                distance[i].append(min(insert, delete, replace))

        return distance[-1][-1]

if __name__ == "__main__":
    sc = SpellChecker()
    sc.update_bow(['dfda', 'fdsa232r'])
