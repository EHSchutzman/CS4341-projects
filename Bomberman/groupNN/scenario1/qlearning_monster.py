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
from qlearning_character import calculate_state

f = open("qtable.txt", "r")

weights = []
qtable = {}

for line in f.readlines():
    line = line.rstrip()
    if not line:
        break
    key = line[:line.find(":")]
    #print(key)
    qtable[key] = line[line.find(":") + 2:]
    #print(line[line.find(":") + 2:])

f.close()

for i in range(0, 100):
    w = open("weights.txt", "r")
    for line in w.readlines():
        line = line.rstrip()
        if not line:
            break
        weights.append(float(line))
    w.close()
    # Create the game
    g = Game.fromfile('map.txt')

    # g.add_monster(StupidMonster("monster",  # name
    #                             "M",  # avatar
    #                             3, 9  # position
    #                             ))

    g.add_monster(SelfPreservingMonster("monster",  # name
                                        "M",  # avatar
                                        3, 13,  # position
                                        2  # detection range
                                        ))

    # TODO Add your character
    q = QCharacter(qtable,   # starting q table
                        weights[0],  # wb
                        weights[1],  # wm
                        weights[2],  # wg
                        weights[3],  # ww
                        weights[4],  # wcm
                        weights[5],  # wcg
                               "Qlearn",  # name
                                "Q",  # avatar
                                0, 0  # position
                                )
    g.add_character(q)
    # Run!
    g.go()
    print(g.world.events)
    print("G DONE::::::: " + str(g.done()))
    print((q.x, q.y))

    w = open("weights.txt", "w")
    weights[0] = q.wb
    weights[1] = q.wm
    weights[2] = q.wg
    weights[3] = q.ww
    for weight in weights:
        w.write(str(weight) + "\n")
    w.close()
    weights.clear()

print(qtable)

f = open("qtable.txt", "w")

keys = qtable.keys()
for k in keys:
    f.write(str(k) + ": " + str(qtable[k]) + "\n")
f.close()