# This is necessary to find the main code
import sys
import operator
import random
import math

sys.path.insert(0, '../bomberman')
# Import necessary stuff
from entity import CharacterEntity
from entity import AIEntity
from entity import MovableEntity
from colorama import Fore, Back, Style, init
from sensed_world import SensedWorld
from events import Event
init(autoreset=True)

gamma = 0.8

class QCharacter(CharacterEntity):

    def __init__(self, qtable, *args, **kwargs):
        super(QCharacter, self).__init__(*args, **kwargs)
        # Whether this character wants to place a bomb
        self.maybe_place_bomb = False
        # Debugging elements
        self.tiles = {}
        self.qtable = qtable

    def do(self, wrld):
        state = calculate_state((self.x, self.y), wrld)
        self.updateQ(state, wrld)

        path = aStar((self.x, self.y), wrld, (7, 18))  # Removed boolean to indicate path, might be good to re-add.

        move = self.select_best_move(state, self.valid_moves(wrld))  # path[len(path) - 1]

        print(self.x, self.y)
        print(wrld.exitcell)
        if (self.x, self.y) == (wrld.exitcell[0], wrld.exitcell[1]):
            print("\n\n WINNNNN \n\n")

        self.move(move[0] - self.x, move[1] - self.y)
        pass

    def select_best_move(self, state, moves):
        candidates = []
        # Construct table keys from possible moves and current state.
        for m in moves:
            candidates.append((state, m))

        m = -1000

        moves = []

        for c in candidates:
            print("Move, score;")
            print(c, self.qtable[c])
            if m < self.qtable[c]:
                moves.clear()
                m = self.qtable[c]
                moves.append(c[1])
            elif m == self.qtable[c]:
                moves.append(c[1])

        return random.choice(moves)

    # Updates the QTable.
    def updateQ(self, state, wrld):
        alpha = 0.3
        moves = get_adjacent((self.x, self.y), wrld)
        keys = self.qtable.keys()
        for m in moves:
            # if q value not initialize, initialize it to 0
            if (state, m) not in keys:
                self.qtable[(state, m)] = 0
            if not wrld.wall_at(m[0], m[1]):
                sim = SensedWorld.from_world(wrld)  # creates simulated world
                c = sim.me(self)  # finds character from simulated world
                c.move(m[0] - self.x, m[1] - self.y)  # moves character in simulated world
                s = sim.next()  # updates simulated world
                c = s[0].me(c)  # gives us character. this is a tuple, we want the board, not the list of elapsed events

                # Check if game is over
                if c is None:
                    print("ENDED!")
                    print(s[0])
                    print(s[1])
                    print("EVENT 0: ")
                    print(s[1][0])
                    for event in s[1]:
                        if event.tpe == Event.CHARACTER_KILLED_BY_MONSTER and event.character.name == self.name:
                            self.qtable[(state, m)] = -5 + gamma * self.qtable[(state, m)]
                        elif event.tpe == Event.CHARACTER_FOUND_EXIT and event.character.name == self.name:
                            self.qtable[(state, m)] = 5 + gamma * self.qtable[(state, m)]
                else:
                    self.qtable[(state, m)] = reward(c, wrld) + gamma * self.qtable[(state, m)]

    def q(self, action):
        # shortsighted update q
        # Q(s, a) = w1*f(s, a) + w2*f(s, a) + w3*f(s, a)
        # ∆ ← [r + γ maxa Q(s, a) − Q(s, a)
        # Q(s, a) ← Q(s, a) + α∆
        # wi ← wi + α∆
        return 0

    def valid_moves(self, wrld):
        moves = get_adjacent((self.x, self.y), wrld)
        final = []
        for m in moves:
            if not wrld.wall_at(m[0], m[1]):
                final.append(m)
        return final

    # Resets styling for each cell. Prevents unexpected/inconsistent behavior that otherwise appears with coloring.
    def reset_cells(self, wrld):
        for x in range(0, wrld.width()):
            for y in range(0, wrld.height()):
                self.set_cell_color(x, y, Fore.RESET + Back.RESET)

    def printWorld(self, wrld):
        w, h = len(wrld.grid), len(wrld.grid[0])
        print('\n\n')
        world = [[0 for x in range(w)] for y in range(h)]

        world[self.y][self.x] = "X"

        for row in world:
            print(row)

# ==================== STATIC METHODS ==================== #
# Many of our methods actually need to be static because they need to be applied to different world state objects.

# Calculates manhattan distance between the two sets of coordinates. These could be tuples, but whatever.
def manhattan_distance(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)


# Prioritizes downward
def cost_to(current, next):
    diff = (next[0] - current[0], next[1] - current[1])
    val = abs(diff[0]) + abs(diff[1])
    if val == 2:
        return 2
    else:
        return 1


# Returns a vector of values representing each feature.
# Vector structure: (bomb distance, monster distance, exit distance)
def calculate_state(coords, wrld):
    return closest_bomb(), closest_monster((coords[0], coords[1]), wrld), euclidean_distance(coords, wrld) + distance_to_exit(coords, wrld)
                            #manhattan_distance(coords[0], coords[1], wrld.exitcell[0], wrld.exitcell[1]) +\



# ==================== FEATURES ==================== #
#   - Distance to closest bomb
#   - Distance to closest monster
#   - 1 / (Distance to exit)^2


# Returns an integer representing the Manhattan distance to the closest bomb.
def closest_bomb():
   return 1  # Will implement this later, in part 2 most likely. For now we don't care about bombs.


# Returns an integer representing the A* distance to the closest monster.
def closest_monster(coords, wrld):
    x = coords[0]
    y = coords[1]
    monsters = monster_tiles(wrld)
    p = float('inf')
    for m in monsters:
        distance = len(aStar((x, y), wrld, m))
        if distance < p:
            p = distance
    return p


# Returns 1/(A* distance to exit)^2.
def distance_to_exit(coords, wrld):
    dist = (len(aStar(coords, wrld, wrld.exitcell)) ** 2)
    if dist < 1:
        return 0
    return 1 / dist

def euclidean_distance(coords, wrld):
    return math.sqrt(((coords[0] - wrld.exitcell[0]) * coords[1] - wrld.exitcell[1]) ** 2)


# Returns a list of tiles which are occupied by at least 1 monster.
def monster_tiles(wrld):
    tiles = []
    for x in range(0, wrld.width()):
        for y in range(0, wrld.height()):
            if wrld.monsters_at(x, y):
                tiles.append((x, y))
    return tiles


# Returns a list of coordinates in the world surrounding the current one.
# param current: An (x, y) point
def get_adjacent(current, wrld):
    # Returns a list of points in the form [(x1, y1), (x2, y2)]
    neighbors = []
    x = current[0]
    y = current[1]

    if x >= 1:
        if y >= 1:
            neighbors.append((x - 1, y - 1))  # top left
        neighbors.append((x - 1, y))  # middle left
        if y < wrld.height() - 1:
            neighbors.append((x - 1, y + 1))  # bottom left

    if y >= 1:
        neighbors.append((x, y - 1))  # top middle
    if y < wrld.height() - 1:
        neighbors.append((x, y + 1))  # bottom middle

    if x < wrld.width() - 1:
        if y >= 1:
            neighbors.append((x + 1, y - 1))  # top right
        neighbors.append((x + 1, y))  # middle right
        if y < wrld.height() - 1:
            neighbors.append((x + 1, y + 1))  # bottom right

    return neighbors

def reward(c, wrld):
    dist = len(aStar((c.x, c.y), wrld, wrld.exitcell))
    if dist == 0:
        return 10
    return 1 / (2 * dist + manhattan_distance(c.x, c.y, wrld.exitcell[0], wrld.exitcell[1]))

def aStar(start, wrld, goal):
    x = start[0]
    y = start[1]
    # print("SELFX: " + str(self.x))
    # print("SELFY: " + str(self.y))
    frontier = []
    frontier.append(((x, y), 0))
    came_from = {}
    cost_so_far = {}
    came_from[(x, y)] = None
    cost_so_far[(x, y)] = 0

    monsters = []


    while not len(frontier) == 0:
        frontier.sort(key=lambda tup: tup[1])  # check that
        current = frontier.pop(0)
        if (current[0][0], current[0][1]) == goal:
            break
        for next in get_adjacent(current[0], wrld):
            if wrld.wall_at(next[0], next[1]):
                cost_so_far[(next[0], next[1])] = 999
                new_cost = 1000
            else:
                new_cost = cost_to(current[0], next) + cost_so_far[current[0]]
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                frontier.append((next, new_cost + manhattan_distance(next[0], next[1], goal[0], goal[1])))
                came_from[next] = current[0]

    cursor = goal
    path = []
    while not cursor == (x, y):
        path.append(cursor)
        try:
            cursor = came_from[cursor]
        except KeyError:
            return 0, 0
    #if draw:
        #print("PATH: ")
        #print(path)

    return path