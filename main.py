import pygame
from pygame.math import Vector2 as Vec2
import sys
import cProfile
import random
from dataclasses import dataclass
from math import e, cos, log10, floor, hypot


# functions
def clamp(value, mn, mx):
    return max(mn, min(value, mx))


def sigfig(x: float, precision: int):
    """
    Rounds a number to number of significant figures
    Parameters:
    - x - the number to be rounded
    - precision (integer) - the number of significant figures
    Returns:
    - float
    """

    x = float(x)
    precision = int(precision)

    return round(x, -int(floor(log10(abs(x)))) + (precision - 1))
  

# classes
@dataclass
class Walk:
    mu: float = 1.40
    sigma: float = 0.36
    min: float = 0.5
    max: float = 2.2


class Pedestrian:
    N = 1
    def __init__(self, x, y):
        self.radius = 5
        self.xvel = clamp(random.gauss(walk.mu, walk.sigma), walk.min, walk.max)
        self.pos = Vec2(x, y)
        self.gate = random.randrange(Gate.N)
        # setup
        self.vel = Vec2(random.uniform(-0.3, 0.3), random.random() * 2 - 1)
        self.acc = Vec2(0, 0)
        self.dacc = Vec2(0, 0)
        self.racc = Vec2(0, 0)
        # driving term
        self.r = all_gates[self.gate].xy
        self.tau = 0.5
        self.v0 = 3
        # repulsive term
    
    def calculate_drive(self):
        self.e = (self.r - self.pos) / (self.r - self.pos).length()
        f = 1 / self.tau * (self.v0 * self.e - self.vel)
        print(self.vel)
        return f
    
    def calculate_repulsion(self, other):
        pass

    def calculate_repulsion_wall(self):
        return Vec2(0, 0)

    def draw(self):
        pygame.draw.aacircle(WIN, pygame.Color("#FDFBD4"), self.pos, self.radius)
        pygame.draw.aacircle(WIN, BLACK, self.pos, 5, 1)
    
    def update(self, i):
        # calculations
        print(i)
        self.acc = Vec2(0, 0)
        self.dacc = Vec2(0, 0)
        self.wacc = Vec2(0, 0)
        # self.racc = Vec2(0, 0)
        #
        self.dacc = self.calculate_drive()
        # self.wacc += self.calculate_repulsion_wall()
        # for other in all_pedestrians:
        #     if other is not self:
        #         self.racc += self.calculate_repulsion(other)
        #
        self.acc = self.dacc + self.racc
        self.vel += self.acc
        self.pos += self.vel
        # rendering
        self.draw()
    

class Gate:
    N = 5
    def __init__(self, x, y):
        self.image = pygame.Surface((20, 60))
        self.image.fill(DARK_GRAY)
        self.rect = self.image.get_frect(topleft=(x, y))
    
    @property
    def xy(self):
        return Vec2(self.rect.midbottom)
    
    def update(self):
        WIN.blit(self.image, self.rect)


# colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
DARK_GRAY = (40, 40, 40)
LIGHT_GRAY = (200, 200, 200)
SMALL = 0.0001

# constants
pygame.init()
WIDTH = 1000
HEIGHT = 700
pygame.display.set_caption("Social Force Model")
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
TARG_FPS = 60
font = pygame.font.SysFont("Courier", 20)


# sim initialization
walk = Walk()
all_pedestrians = []
all_gates = []
for i in range(Gate.N):
    gate = Gate(i * 180 + 140, 50)
    all_gates.append(gate)
for _ in range(Pedestrian.N):
    ped = Pedestrian(WIDTH / 2 + random.randint(-200, 200), HEIGHT - 50)
    all_pedestrians.append(ped)


def main():
    # mainloop
    running = __name__ == "__main__"
    while running:
        clock.tick(5)
    
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
    
        # clearing window
        WIN.fill(LIGHT_GRAY)

        # rendering the scene
        for gate in all_gates:
            gate.update()

        for i, ped in enumerate(all_pedestrians):
            ped.update(i)
    
        # flip the display
        pygame.display.flip()

    pygame.quit()
    sys.exit()


main()