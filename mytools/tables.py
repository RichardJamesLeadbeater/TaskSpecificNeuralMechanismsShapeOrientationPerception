import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import table
import matplotlib as mpl


def create_table(df, font_size=5, header_font_size=None, auto_width=False,
                 save_path=None, dpi=300, col_width=0.5, col_height=0.3, _close=True, _show=False):
    if header_font_size is None:
        header_font_size = font_size
    # Initialise font
    mpl.rcParams['font.family'] = 'Arial'
    # Create a Table Plot
    fig, ax = plt.subplots(figsize=(6, 3))  # Adjust the figsize as needed
    # fig, ax = plt.subplots()  # Adjust the figsize as needed
    # Hide axis
    ax.axis('off')
    # Create the table plot
    tbl = table(ax, df, loc='center', cellLoc='center', colWidths=[col_width] * len(df.columns))
    tbl.auto_set_font_size(False)  # needs to be set to change font size
    tbl.set_fontsize(font_size)

    if auto_width is True:
        [tbl.auto_set_column_width(i) for i in range(len(df.columns))]
        [tbl.auto_set_column_width(False) for _ in df.columns]

    cell_dict = tbl.get_celld()
    for cell_key in cell_dict:
        i_cell = cell_dict[cell_key]
        # Font size
        if cell_key[0] == 0:  # Header row
            i_cell.set_text_props(fontweight='bold', fontsize=header_font_size)  # Adjust font properties
        # Remove index column
        if cell_key[1] == -1:
            cell_dict[cell_key].visible_edges = ''
            cell_dict[cell_key].set_visible(False)
    # Save the Image with the specified pad_inches
    plt.axis('tight')  # Remove excess white space
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(f"{save_path}.png", dpi=dpi)  # Adjust the filename and DPI as needed
    # Display the Image (Optional)
    if _close is True:
        plt.close()
    else:
        if _show is True:
            plt.show()
    return fig, ax

