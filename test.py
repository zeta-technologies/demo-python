from pygame import *
perso = image.load('Dragon Rouge.png')
scr = display.set_mode((500,500))
class Perso:
    image = dict([(direction,[perso.subsurface(x,y,96,96)for x in range(0,384,96)]) for direction,y in zip((K_DOWN,K_LEFT,K_RIGHT,K_UP),range(0,384,96))])
    x,y = 202,202

direction = K_DOWN
index_img = 0

scr.blit(Perso.image[direction][index_img],(202,202))
display.flip()

while True:
    ev = event.poll()
    if ev.type == QUIT: break
    k = key.get_pressed()
    for i in (K_DOWN,K_LEFT,K_RIGHT,K_UP):
        if k[i]:
            direction = i if direction != i else direction
            index_img = (index_img+1)%4
            Perso.x += (-k[K_LEFT]+k[K_RIGHT])*8
            Perso.y += (-k[K_UP]+k[K_DOWN])*8
            break
    else:
        index_img = 0

    scr.fill((50,50,50),special_flags=BLEND_RGB_SUB) # ajoute un effet de trainée, sinon mettre scr.fill(0)
    scr.blit(Perso.image[direction][index_img],(Perso.x,Perso.y))
    display.flip()
    time.wait(50)
