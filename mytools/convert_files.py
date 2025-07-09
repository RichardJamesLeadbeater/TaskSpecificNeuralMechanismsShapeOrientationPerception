from fontTools.ttLib import TTFont


def convert_otf_to_ttf(otf_filepath):
    # Open the OTF font file and convert to TTF
    font = TTFont(otf_filepath)

    # Save it as a TTF file
    ttf_filepath = f"{otf_filepath[:-4]}.ttf"  # Replace .otf with .ttf
    font.save(ttf_filepath)
