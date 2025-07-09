import os
from tkinter import filedialog, Tk


def browser(get_file=False, get_dir=True, chdir=False):
    root = Tk()  # Create tkinter window

    root.withdraw()  # Hide tkinter window
    root.update()

    if get_file:
        file = filedialog.askopenfile()
    elif get_dir:
        directory = filedialog.askdirectory()

    root.update()
    root.destroy()  # Destroy tkinter window

    if chdir:
        os.chdir(directory)

    if get_file:
        return file
    if get_dir:
        return directory
