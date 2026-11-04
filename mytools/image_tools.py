from PIL import Image, ImageDraw, ImageFont
import string


def stitch_images(image_files, output_file, direction="vertical", match_size=True,
                  separator=True, separator_size=3,
                  separator_colour=(150, 150, 150), panel_gap=15,
                  labels=None, label_pad=40):

    imgs = [
        trim_vertical_whitespace(Image.open(f).convert("RGB"))
        for f in image_files
    ]

    if match_size:
        if direction == "vertical":
            target = max(img.width for img in imgs)
            imgs = [
                img if img.width == target else
                img.resize((target, round(img.height * target / img.width)), Image.LANCZOS)
                for img in imgs
            ]
        elif direction == "horizontal":
            target = max(img.height for img in imgs)
            imgs = [
                img if img.height == target else
                img.resize((round(img.width * target / img.height), target), Image.LANCZOS)
                for img in imgs
            ]
        else:
            raise ValueError("direction must be 'vertical' or 'horizontal'")

    if labels is True:
        labels = list(string.ascii_uppercase)

    gap_before = panel_gap // 2
    gap_after = panel_gap - gap_before
    sep_space = gap_before + gap_after + (separator_size if separator else 0)

    left_pad = label_pad if (labels and direction == "vertical") else 0
    top_pad = label_pad if (labels and direction == "horizontal") else 0

    if direction == "vertical":

        max_width = max(img.width for img in imgs)
        total_height = sum(img.height for img in imgs) + (len(imgs) - 1) * sep_space

        font_size = round(total_height * 0.04)
        font = ImageFont.truetype("arial.ttf", font_size)

        left_pad = round(max(left_pad, font_size))

        combined = Image.new("RGB", (max_width + left_pad, total_height), "white")
        draw = ImageDraw.Draw(combined)

        y = 0
        for i, img in enumerate(imgs):

            combined.paste(img, (left_pad, y))

            if labels:
                label = labels[i]
                draw.text((10, y + 10), label, fill="black", font=font)
            y += img.height

            if i < len(imgs) - 1:
                y += gap_before

                if separator:
                    draw.rectangle(
                        [left_pad, y,
                         left_pad + max_width - 1,
                         y + separator_size - 1],
                        fill=separator_colour
                    )
                    y += separator_size

                y += gap_after

    else:

        max_height = max(img.height for img in imgs)
        total_width = sum(img.width for img in imgs) + (len(imgs) - 1) * sep_space

        font_size = round(max_height * 0.04)
        font = ImageFont.truetype("arial.ttf", font_size)

        top_pad = round(max(top_pad, font_size * 1.02))

        combined = Image.new("RGB", (total_width, max_height + top_pad), "white")
        draw = ImageDraw.Draw(combined)

        x = 0
        for i, img in enumerate(imgs):

            combined.paste(img, (x, top_pad))

            if labels:
                label = labels[i]
                bbox = draw.textbbox((0, 0), label, font=font)
                w = bbox[2] - bbox[0]
                draw.text((x + (img.width - w) // 2, 10),
                          label, fill="black", font=font)
            x += img.width

            if i < len(imgs) - 1:
                x += gap_before

                if separator:
                    draw.rectangle(
                        [x, top_pad,
                         x + separator_size - 1,
                         top_pad + max_height - 1],
                        fill=separator_colour
                    )
                    x += separator_size

                x += gap_after

    combined.save(output_file)
    return combined


def trim_vertical_whitespace(img, threshold=250):

    gray = img.convert("L")

    top = 0
    while top < img.height:
        if min(gray.crop((0, top, img.width, top + 1)).getdata()) < threshold:
            break
        top += 1

    bottom = img.height
    while bottom > top:
        if min(gray.crop((0, bottom - 1, img.width, bottom)).getdata()) < threshold:
            break
        bottom -= 1

    return img.crop((0, top, img.width, bottom))