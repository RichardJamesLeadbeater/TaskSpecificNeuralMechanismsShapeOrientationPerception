import pyglet
from psychopy import visual


def create_window(**kwargs):
    # Set up the Window
    which_screen = 0
    fullscreen = True
    screen = pyglet.canvas.Display().get_default_screen()  # get screen dimensions
    my_win = visual.Window(
        size=(screen.width, screen.height), fullscr=fullscreen, screen=which_screen,
        winType='pyglet', allowGUI=False, allowStencil=False,
        monitor=kwargs.get('monitor'), gamma=1, color=kwargs.get('bg_color', [0.5] * 3), colorSpace='rgb',
        blendMode='avg', useFBO=True, units='norm',
    )
    my_win.setMouseVisible(False)
    return my_win
