# This is necessary to find the main code
import sys
import math

sys.path.insert(0, '../bomberman')
# Import necessary stuff
from entity import CharacterEntity
from colorama import Fore, Back, Style, init
init(autoreset=True)

class TestCharacterExpecti(CharacterEntity):

    def expectiMax(self, wrld):

        return

    def expVal(self, wrld, max_depth):
        value = 0
        prob = 0
        if max_depth == 0:
            return value
        actions = []
        for a in actions:
            prob = self.probability(a)
            value += prob * value(a)


    

    def maxVal(self, wrld, max_depth):
        value = float("-inf")
        if max_depth == 0:
            return value

        actions = [] #get available actions that player can do
        for a in actions:
            value = max(value, self.expVal(a, max_depth -1))
        return value

    def probability(self, wrld):
        return
