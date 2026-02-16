ImageModify - Project Overview
==============================

Python Files
------------

ReduceImageSize.py
    Resizes all images in a given folder by a specified percentage.
    It takes two command-line arguments: a folder path and a percentage value.
    A percentage below 100 reduces image dimensions, above 100 enlarges them.
    Resized images are saved to a subfolder named "Reduced_<YYYYMMDD>" inside
    the source folder. Original images are never modified.

    Supported formats: JPG, JPEG, PNG, BMP, GIF, TIFF, WEBP

    Usage:
        python ReduceImageSize.py <folder_path> <percentage>

    Examples:
        python ReduceImageSize.py c:\shared 50     (shrink to 50%)
        python ReduceImageSize.py c:\shared 200    (enlarge to 200%)
