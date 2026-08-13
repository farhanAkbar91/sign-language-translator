@echo off
if not exist .venv (
    echo [ERROR] Virtual Environment .venv belum ada!
    echo Menjalankan setup_env.bat terlebih dahulu...
    call setup_env.bat
)
.venv\Scripts\python train_model.py
pause
