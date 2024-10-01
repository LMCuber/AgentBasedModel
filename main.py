import pygame
from pygame.math import Vector2 as Vec2
import sys
import cProfile
import random
from dataclasses import dataclass
from math import e, cos, log10, floor, exp
from enum import Enum


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


files = open("file.txt", "w")


class Pedestrian:
    N = 200
    passed_right = 0
    passed_left = 0
    def __init__(self, x, y, color=None):
        # init
        self.passed = False
        self.pass_right = x == 50
        self.radius = 5
        self.pos = Vec2(x, y)
        self.gate = random.randrange(Gate.N)
        self.dest = Vec2(WIDTH - 50, HEIGHT / 2)
        self.color = color if color is not None else pygame.Color(pygame.Color("#FDFBD4"))
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
        self.r0 = 4
        self.B = 3
    
    def calculate_drive_force(self):
        self.e = (self.dest - self.pos) / (self.dest - self.pos).length()
        desired_vel = self.v0 * self.e
        delta_vel = desired_vel - self.vel
        return 1 / self.t * delta_vel
    
    def calculate_obstacle_force(self):
        total_f = Vec2(0, 0)
        for ob in all_obstacles:
            n, d = ob.get_distance(self)
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
        #
        if not self.passed:
            if self.pass_right and self.pos.x >= WIDTH / 2:
                self.passed = True
                Pedestrian.passed_right += 1
            if not self.pass_right and self.pos.x <= WIDTH / 2:
                self.passed = True
                Pedestrian.passed_left += 1
        # draw
        self.draw()
    
    def draw(self):
        pygame.draw.aacircle(WIN, self.color, self.pos, self.radius)
        pygame.draw.aacircle(WIN, BLACK, self.pos, 5, 1)
        #
        m = 3
        pygame.draw.line(WIN, (0, 255, 0), self.pos, self.pos + self.vel * m, 2)
        pygame.draw.line(WIN, (0, 0, 255), self.pos, self.pos + self.acc * m, 2)
    

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
    def update(self):
        WIN.blit(self.image, self.rect)


class Obstacle(AbstractObstacle):
    def __init__(self, x, y):
        self.image = pygame.Surface((24, 24), pygame.SRCALPHA)
        # pygame.draw.rect(self.image, BLACK, (0, 0, *self.image.size), 2)
        pygame.draw.circle(self.image, BLACK, [s / 2 for s in self.image.size], self.image.width / 2, 2)
        self.rect = self.image.get_frect(topleft=(x, y))
    
    @property
    def xy(self):
        return Vec2(self.rect.center)
    
    def get_distance(self, other):
        normal = (other.pos - self.xy)
        return (normal, normal.length())
    

class Wall(AbstractObstacle):
    def __init__(self, p1, p2):
        self.p1 = Vec2(p1) 
        self.p2 = Vec2(p2)
        self.w = p2[0] - p1[0] + 1
        self.h = p2[1] - p1[1] + 1
        self.image = pygame.Surface((self.w, self.h))
        pygame.draw.line(self.image, BLACK, self.p1, self.p2)
        self.rect = self.image.get_rect(topleft=p1)
    
    @property
    def xy(self):
        return Vec2(self.rect.center)
    
    def get_distance(self, other):
        line_vec = self.p2 - self.p1
        pnt_vec = other.pos - Vec2(self.p1)
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
        nearest = nearest + self.p1
        #
        normal = (other.pos - nearest)
        # pygame.draw.line(WIN, (255, 0, 0), nearest, other.pos)
        return (normal, dist)
    

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

gap = 80
y = 0
while True:
    p1 = (WIDTH / 2, y)
    y += random.randint(30, 100)
    p2 = (WIDTH / 2, y)
    ob = Wall(p1, p2)
    all_obstacles.append(ob)
    y += gap
    if y >= HEIGHT:
        break


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
            WIN.blit(ob.image, ob.rect)
        
        surf = font.render(str(Pedestrian.passed_left), True, BLACK)
        WIN.blit(surf, (WIDTH - 50, HEIGHT - 60))
        surf = font.render(str(Pedestrian.passed_right), True, BLACK)
        WIN.blit(surf, (50, HEIGHT - 60))

        surf = font.render(str(int(clock.get_fps())), True, BLACK)
        WIN.blit(surf, (10, 10))
    
        # flip the display
        pygame.display.flip()

    pygame.quit()
    sys.exit()

main()