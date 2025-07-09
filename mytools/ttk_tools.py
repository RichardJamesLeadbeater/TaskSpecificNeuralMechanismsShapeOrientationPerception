from tkinter import ttk


def lift_window(window):
    """Force window to front"""
    window.lift()
    window.focus_force()
    window.attributes('-topmost', True)


def lower_window(window):
    """Move window to back"""
    window.attributes('-topmost', False)
    window.lower()


from tkinter import ttk


def labelled_entry(parent, label_text, style=None, default_entry=""):
    # Fallback to default if style not provided
    if style is None:
        style = "TLabel"

    # Create the frame
    frame = ttk.Frame(parent)

    # Apply label style: If a style is passed, use it, otherwise use font
    label = ttk.Label(frame, text=label_text)
    label.config(style=style)
    label.pack()

    # Add space after label
    add_blank_space(frame, pady=(0, 0))

    # Create the entry widget
    entry = ttk.Entry(frame, style=style)  # Default
    entry.insert(0, default_entry)  # Insert default value
    entry.pack()

    return frame, entry


def dropdown_menu(parent, label_text="", values=(), default_idx=0,
                  combo_style=None, label_style=None):
    # Apply defaults
    if combo_style is None:
        combo_style = "TCombobox"
    if label_style is None:
        label_style = "TLabel"
    # Create frame
    frame = ttk.Frame(parent)
    # Add label
    label = ttk.Label(frame, text=label_text)
    label.config(style=label_style)
    label.pack()
    # Add space
    add_blank_space(frame, pady=(1, 1))
    # Add dropdown menu
    combo = ttk.Combobox(frame, values=values)
    combo.config(style=combo_style)
    combo.current(default_idx)  # Default value
    combo.pack()

    return frame, combo


def set_state_all_widgets(frame, state):
    # Loop through parent widgets
    for i_widget in frame.winfo_children():
        if 'state' in i_widget.keys():
            i_widget.configure(state=state)  # Set state
        # Recurse into child widgets
        elif isinstance(i_widget, ttk.Frame):
            set_state_all_widgets(i_widget, state)


def add_blank_space(my_frame, pady=(0, 0), font=("Helvetica", 1)):
    """Adds blank space to the given frame with the specified height."""
    blank_space = ttk.Label(my_frame, text="", font=font)  # Create an empty label
    blank_space.pack(pady=pady)  # Add blank space with specified height


def bind_entries(parent, commands):
    """
    Recursively binds multiple functions to the '<KeyRelease>' event of all ttk.Entry widgets under a parent widget.

    Parameters:
        parent (tk.Widget): The parent widget from which to start searching for ttk.Entry widgets.
        commands (list): A list of functions to be called whenever a key is released in any Entry widget.
                         Each function should accept a single argument (the event object).
    """
    def combined_commands(event):
        for cmd in commands:
            cmd(event)

    # Iterate through all child widgets of the parent
    for child in parent.winfo_children():
        # If the child is a ttk.Entry widget
        if isinstance(child, ttk.Entry):
            # Bind each function to the <KeyRelease> event of the Entry
            child.bind("<KeyRelease>", combined_commands)
        else:
            # If the child is a container (e.g., Frame), apply the same binding recursively
            bind_entries(child, commands)


def list_style_options():
    style_options = {
        "font": "Set the font of the widget (e.g., ('Helvetica', 12))",
        "foreground": "Set the text color (e.g., 'blue', 'red', '#ff0000')",
        "background": "Set the background color (e.g., 'white', 'green', '#00ff00')",
        "padding": "Set the padding around the widget (e.g., 5, (5, 10), (10, 20, 10, 20))",
        "relief": "Set the relief (appearance) of the widget (e.g., 'flat', 'raised', 'sunken', 'groove')",
        "width": "Set the width of the widget (integer number of characters)",
        "height": "Set the height of the widget (integer number of lines)",
        "anchor": "Set the anchor position of the widget's content (e.g., 'w', 'center', 'e', 'ne')",
        "justify": "Set how text is aligned within the widget (e.g., 'left', 'center', 'right')",
        "wraplength": "Set the maximum width (in pixels) for text wrapping (e.g., 200, 300)",
        "borderwidth": "Set the width of the widget's border (integer)",
        "highlightthickness": "Set the thickness of the widget's focus highlight (integer)",
        "highlightcolor": "Set the color of the focus highlight (e.g., 'red')",
        "highlightbackground": "Set the background color of the focus highlight (e.g., 'black')",
        "state": "Set the state of the widget (e.g., 'normal', 'disabled', 'active')",
        "image": "Set an image to be displayed in the widget (e.g., a PhotoImage object)",
        "padx": "Set horizontal padding (e.g., 5, 10)",
        "pady": "Set vertical padding (e.g., 5, 10)",
        "takefocus": "Set whether the widget should take focus (True or False)"
    }

    print("List of common style options in ttk.Style().configure():\n")
    for option, description in style_options.items():
        print(f"{option}: {description}\n")

