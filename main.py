import pygame
from pygame.math import Vector2 as Vec2
import sys
import cProfile
import random
from dataclasses import dataclass
from math import e, cos, sin, log10, floor, exp, pi
from enum import Enum


# functions
def dist_point_to_line_segment(p1, p2, pos):
    line_vec = p2 - p1
    pnt_vec = pos - Vec2(p1)
    line_len = line_vec.length()
    line_unitvec = line_vec.normalize()
    pnt_vec_scaled = pnt_vec / line_len
    t = line_unitvec.dot(pnt_vec_scaled)    
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    nearest = line_vec * t
    dist = (nearest - pnt_vec).length()
    nearest = nearest + p1
    #
    normal = (pos - nearest)
    return (normal, dist)

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


files = open("file.txt", "w")


class Pedestrian:
    N = 50
    def __init__(self, x, y, color=None):
        # init
        self.radius = 5
        self.pos = Vec2(x, y)
        self.gate = random.randrange(Gate.N)
        self.dest = Vec2(WIDTH - 50, HEIGHT / 2)
        self.color = color if color is not None else pygame.Color("#FDFBD4")

        # driving term
        self.v0 = clamp(random.gauss(walk.mu, walk.sigma), walk.min, walk.max)*2
        self.v0 = 3
        self.vel = Vec2(0, 0)
        self.acc = Vec2(0, 0)
        self.dacc = Vec2(0, 0)
        self.oacc = Vec2(0, 0)
        self.iacc = Vec2(0, 0)
        self.t = 10
        # repulsive term
        self.A = 5
        self.B = 0.1
        self.labda = 0.4
        # obstacle term
        self.r0 = 5
        # interactive term
        self.B = 4
    
    def calculate_drive_force(self):
        self.e = (self.dest - self.pos) / (self.dest - self.pos).length()
        desired_vel = self.v0 * self.e
        delta_vel = desired_vel - self.vel
        return 1 / self.t * delta_vel
    
    def calculate_obstacle_force(self):
        total_f = Vec2(0, 0)
        for ob in all_obstacles:
            for n, d in ob.get_distances(self):
                f = exp(-d / self.r0) * n
                total_f += f
        return total_f
    
    def calculate_interactive_force(self):
        total_f = Vec2(0, 0)
        for ped in all_pedestrians:
            if ped is not self:
                d = (ped.pos - self.pos).length()
                n = (self.pos - ped.pos)
                f = exp(-d / self.B) * n
                total_f += f
        return total_f
    
    def update(self, dt):
        # update
        self.dacc = self.calculate_drive_force()
        self.oacc = self.calculate_obstacle_force()
        self.iacc = self.calculate_interactive_force()
        # newton
        self.acc = self.dacc + self.oacc + self.iacc
        self.vel += self.acc
        self.pos += self.vel * dt
        # draw
        self.draw()
    
    def draw(self):
        pygame.draw.aacircle(WIN, self.color, self.pos, self.radius)
        pygame.draw.aacircle(WIN, BLACK, self.pos, 5, 1)
        #
        m = 5
        pygame.draw.line(WIN, (0, 255, 0), self.pos, self.pos + self.vel * m, 2)
        pygame.draw.line(WIN, (255, 140, 0), self.pos, self.pos + self.acc * m * 7, 2)
    

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


class AbstractObstacle:
    def draw(self):
        WIN.blit(self.image, self.rect)


class Obstacle(AbstractObstacle):
    def __init__(self, x, y):
        self.image = pygame.Surface((24, 24), pygame.SRCALPHA)
        # pygame.draw.rect(self.image, BLACK, (0, 0, *self.image.size), 2)
        pygame.draw.circle(self.image, BLACK, [s / 2 for s in self.image.size], self.image.width / 2, 2)
        self.rect = self.image.get_frect(topleft=(x, y))
    
    def update(self):
        self.draw()
    
    @property
    def xy(self):
        return Vec2(self.rect.center)
    
    def get_distances(self, other):
        normal = (other.pos - self.xy)
        return [(normal, normal.length())]
    

class Polygon(AbstractObstacle):
    def __init__(self, *points):
        self.points = [Vec2(p) for p in points]
    
    def update(self):
        self.draw()
    
    def draw(self):
        pygame.draw.lines(WIN, BLACK, True, self.points)
    
    def get_distances(self, other):
        for i in range(len(self.points)):
            p1 = self.points[i]
            try:
                p2 = self.points[i + 1]
            except IndexError:
                p2 = self.points[0]
            yield dist_point_to_line_segment(p1, p2, other.pos)


class Revolver(AbstractObstacle):
    def __init__(self, p1, n, l, av):
        self.p1 = Vec2(p1)
        self.n = n
        self.l = l
        self.av = av
        self.angle = 0
        self.p2s = [Vec2(0, 0) for _ in range(self.n)]
    
    def update(self):
        self.angle += self.av
        for i in range(self.n):
            a = self.angle + i * (2 * pi) / self.n
            arm = self.l * Vec2(cos(a), sin(a))
            p2 = self.p1 + arm
            self.p2s[i] = p2
            pygame.draw.line(WIN, BLACK, self.p1, p2)
            
    def get_distances(self, other):
        for p2 in self.p2s:
            yield dist_point_to_line_segment(self.p1, p2, other.pos)

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
all_obstacles = []

for _ in range(Pedestrian.N):
    ped = Pedestrian(50, random.randrange(HEIGHT), pygame.Color("#0F4C5C"))
    all_pedestrians.append(ped)

all_obstacles.append(Revolver((WIDTH / 2, HEIGHT / 2), 4, 80, 0.02))
xo = 300
yo = 100
all_obstacles.append(Polygon(
    (WIDTH / 2 - xo, HEIGHT / 2 - yo),
    (WIDTH / 2 + xo, HEIGHT / 2 - yo),
))
all_obstacles.append(Polygon(
    (WIDTH / 2 + xo, HEIGHT / 2 + yo),
    (WIDTH / 2 - xo, HEIGHT / 2 + yo),
))



def main():
    # mainloop
    running = __name__ == "__main__"
    while running:
        dt = clock.tick(TARG_FPS) / 1000 / (1 / 120)
    
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
            ped.dest = Vec2(pygame.mouse.get_pos())
            ped.update(1)
        for ob in all_obstacles:
            ob.update()
        
        surf = font.render(str(int(clock.get_fps())), True, BLACK)
        WIN.blit(surf, (10, 10))
    
        # flip the display
        pygame.display.flip()

    pygame.quit()
    sys.exit()

main()