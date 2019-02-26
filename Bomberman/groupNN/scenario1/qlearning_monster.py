# The sole purpose of this class is to be able to carry q-table over between iterations of the game.

# This is necessary to find the main code
import sys
sys.path.insert(0, '../../bomberman')
sys.path.insert(1, '..')

# Import necessary stuff
from game import Game

# TODO This is your code!
sys.path.insert(1, '../groupNN')
from qlearning_character import QCharacter
from monsters.stupid_monster import StupidMonster
from monsters.selfpreserving_monster import SelfPreservingMonster

f = open("qtable.txt", "r")

qtable = {}

for line in f.readlines():
    line = line.rstrip()
    if not line:
        break
    key = line[:line.find(":")]
    print(key)
    qtable[key] = line[line.find(":") + 2:]
    print(line[line.find(":") + 2:])

f.close()

for i in range(0, 50):
    # Create the game
    g = Game.fromfile('simpleworld.txt')

    # g.add_monster(StupidMonster("monster",  # name
    #                             "M",  # avatar
    #                             3, 9  # position
    #                             ))

    # g.add_monster(SelfPreservingMonster("monster",  # name
    #                                     "M",  # avatar
    #                                     3, 13,  # position
    #                                     2  # detection range
    #                                     ))

    # TODO Add your character
    g.add_character(QCharacter(qtable,  # starting q table
                               "Qlearn",  # name
                                "Q",  # avatar
                                0, 0  # position
                                ))
    # Run!
    g.go()

print(qtable)

f = open("qtable.txt", "w")


keys = qtable.keys()
for k in keys:
    f.write(str(k) + ": " + str(qtable[k]) + "\n")
f.close()
