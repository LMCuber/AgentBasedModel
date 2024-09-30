import pygame
from pygame.math import Vector2 as Vec2
import sys
import cProfile
import random
from dataclasses import dataclass
from math import e, cos, log10, floor


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
    N = 20
    def __init__(self, x, y):
        self.radius = 5
        self.xvel = clamp(random.gauss(walk.mu, walk.sigma), walk.min, walk.max)
        self.pos = Vec2(x, y)
        self.gate = random.randrange(Gate.N)
        self.dest = all_gates[self.gate].xy
        # driving term
        self.v0 = 10
        self.vel = Vec2(random.uniform(-0.3, 0.3), random.random() * 2 - 1)
        self.acc = Vec2(0, 0)
        self.dacc = Vec2(0, 0)
        self.raccs = Vec2(0, 0)
        self.t = 10
        # repulsive term
        self.A = 5
        self.B = 0.1
        self.labda = 0.4
    
    def calculate_drive(self):
        self.e = (self.dest - self.pos).normalize()
        desired_vel = self.v0 * self.e
        delta_vel = desired_vel - self.vel
        self.dacc = 1 / self.t * delta_vel
        print("drive", self.dacc)
    
    def calculate_repulsion(self, other):
        r_ab = self.radius + other.radius
        d_ab = (other.pos - self.pos).length()
        normal = (other.pos - self.pos).normalize()
        cosphi = -normal * self.e
        # f = self.A * e ** ((r_ab - d_ab) / (self.B)) * normal * (self.labda + (1 - self.labda) * ((1 + cosphi) / 2))
        dx = other.pos.x - self.pos.x
        dy = other.pos.y - self.pos.y
        if d_ab < 20:
            self.raccs += Vec2(-dx, -dy)

    def draw(self):
        pygame.draw.aacircle(WIN, pygame.Color("#FDFBD4"), self.pos, self.radius)
        pygame.draw.aacircle(WIN, BLACK, self.pos, 5, 1)
        #
        repr_f = ", ".join([str(sigfig(x, 2)) for x in self.raccs])
        surf = font.render(repr_f, True, BLACK)
        WIN.blit(surf, self.pos)
    
    def newton(self):
        self.acc = self.dacc + self.raccs
        self.vel += self.acc
        speed = self.vel.length()
        if speed > 1:
            self.vel = self.vel / speed
        self.pos += self.vel
    
    def update(self):
        # init
        self.dest = pygame.mouse.get_pos()
        self.raccs = Vec2(0, 0)
        # calculations
        self.calculate_drive()
        for other in all_pedestrians:
            if other is not self:
                self.calculate_repulsion(other)
        self.newton()
        # rendering
        try:
            self.draw()
        except Exception:
            pass
    

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
        clock.tick(TARG_FPS)
    
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


        for ped in all_pedestrians:
            ped.update()
    
        # flip the display
        pygame.display.flip()

    pygame.quit()
    sys.exit()


main()