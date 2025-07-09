from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_TABLE_DIRECTION, WD_CELL_VERTICAL_ALIGNMENT, WD_ROW_HEIGHT_RULE
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls
from docx.shared import Cm, Pt
import os


def create_table(data, save_path, _return_table=True):
    # EXAMPLE DATA FORMAT
    # table_data = [
    #     ['Name', 'Age', 'Occupation'],
    #     ['John Doe', '30', 'Engineer'],
    #     ['Jane Smith', '25', 'Designer'],
    #     ['Bob Johnson', '40', 'Manager']
    # ]

    num_rows = len(data)
    num_cols = len(data[0])

    doc = Document()
    table = doc.add_table(rows=num_rows, cols=num_cols)

    # Customize table style
    table.style = 'Table Grid'

    # params
    cell_width = Cm(3)
    row_height = Cm(1.5)

    # Set cell width (change as per your requirement)
    for col in table.columns:
        for cell in col.cells:
            cell.width = cell_width

    # Set cell alignment
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.direction = WD_TABLE_DIRECTION.LTR
    for row in table.rows:
        row.height = row_height
        for cell in row.cells:
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER

    # Populate the table with data
    for i, row_data in enumerate(data):
        for j, cell_data in enumerate(row_data):
            table.cell(i, j).text = str(cell_data)

    # Save the document
    save_path = f"{save_path}_table.docx"
    doc.save(os.path.join(save_path))

    if _return_table is True:
        return table

