# EMOTIVA Typography Assets

This directory is designated for local and commercial font files used by EMOTIVA:

1. **BD Supper** (Header & Display Typography):
   - Designed by Büro Destruct (Typedifferent).
   - Commercial font. If you hold a licensed `.woff2` or `.ttf` copy:
     - Place `BDSupper.woff2` or `BDSupper.ttf` in this `fonts/` folder, OR
     - Install `BD Supper` directly on your Windows operating system (Control Panel -> Fonts).
   - The `@font-face` declaration in `app.py`, `src/visualization.py`, and `src/report.py` will automatically detect and render it via `local('BD Supper')`, `local('BDSupper')`, and `url('fonts/BDSupper.woff2')`.

2. **Forum** (Body & Interface Typography):
   - Open-source Google Font.
   - Automatically loaded via Google Fonts CDN (`https://fonts.googleapis.com/css2?family=Forum&display=swap`).
