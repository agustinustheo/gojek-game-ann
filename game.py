import pygame

pygame.init()

black = (0,0,0)
white = (255,255,255)
red = (255,0,0)
green = (0,255,0)
blue = (0,0,255)

display_width=267
display_height=477
padding=10
gameDisplay = pygame.display.set_mode((display_width, display_height))
pygame.display.set_caption('A bit Racey')
clock = pygame.time.Clock()

playerImg = pygame.image.load('objects/player.png')
player_img_size = playerImg.get_rect().size
def player(x, y):
    gameDisplay.blit(playerImg, (x, y))

def game_loop():

    x = (display_width * 0.45)
    y = (display_height * 0.8)

    x_change = 0

    crashed = False
    while not crashed:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                crashed = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    x_change=-50
                elif event.key == pygame.K_RIGHT:
                    x_change=50
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                    x_change=0

        # Create boundary
        if x+x_change <= display_width-padding-player_img_size[0] and x+x_change >= padding:
            x+=x_change

        gameDisplay.fill(white)
        player(x, y)

        pygame.display.update()
        clock.tick(30)

game_loop()
pygame.quit()
quit()