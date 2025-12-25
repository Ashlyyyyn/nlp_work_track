from matplotlib import font_manager

# Force Matplotlib to regenerate its cached font list.
font_manager._load_fontmanager(try_read_cache=False)
print("Matplotlib font cache rebuilt.")