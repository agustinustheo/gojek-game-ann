import pygame

pygame.init()

black = (0,0,0)
white = (255,255,255)
red = (255,0,0)
green = (0,255,0)
blue = (0,0,255)
gray = (141,139,160)

display_width=280
display_height=449
padding=41
gameDisplay = pygame.display.set_mode((display_width, display_height))
pygame.display.set_caption('A bit Racey')
clock = pygame.time.Clock()

playerImg = pygame.image.load('objects/player.png')
player_img_size = playerImg.get_rect().size

background = pygame.image.load('objects/background.png')
obs1 = pygame.image.load('objects/obstacle1.jpg')
obs2 = pygame.image.load('objects/obstacle2.jpg')

playerPos = 0

def player(x, y):
    gameDisplay.blit(playerImg, (x, y))

def game_loop():

    x = 70#(display_width * 0.45)
    y = display_height-86#(display_height * 0.8)

    bgY = 0 #posisi obs
    bgY2 = -350 #posisi obs

    x_change = 0

    crashed = False
    while not crashed:
        clock.tick(15)

        bgY += 10  # Move both background images back
        bgY2 += 10

        if bgY > background.get_height() :  #reset obs kiri
            bgY = -1000

        if bgY2 > background.get_height() : #reset obs kanan
            bgY2 = -320

        if bgY > display_height-190 and playerPos == 0:  #obs kiri collide
            crashed = True

        if bgY2 > display_height-190 and playerPos == 1:  #obs kanan collide
            crashed = True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                crashed = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    x_change=-92
                    playerPos = 0
                elif event.key == pygame.K_RIGHT:
                    x_change=92
                    playerPos = 1
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                    x_change=0

        # Create boundary
        if x+x_change <= display_width-padding-player_img_size[0] and x+x_change >= padding:
            x+=x_change
        
        gameDisplay.fill(white) #or gray
        # gameDisplay.blit(background,(0,0))
        gameDisplay.blit(obs1,(70, bgY))
        gameDisplay.blit(obs2,(162, bgY2))
        player(x, y)

        pygame.display.update()

game_loop()
pygame.quit()
quit()