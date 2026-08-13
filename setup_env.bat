@echo off
echo ===================================================
echo   Membuat dan Menyiapkan Virtual Environment (.venv)
echo ===================================================

py -3.11 --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python 3.11 tidak ditemukan!
    echo Silakan install Python 3.11 dari https://www.python.org/downloads/release/python-3119/
    pause
    exit /b 1
)

if not exist .venv (
    echo Membuat environment .venv dengan Python 3.11...
    py -3.11 -m venv .venv
)

echo Menginstall / Memperbarui library dependensi...
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\pip install -r requirements.txt

echo.
echo ===================================================
echo   SETUP SELESAI! Anda siap menjalankan proyek.
echo ===================================================
pause
