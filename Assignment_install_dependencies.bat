@echo off
echo =====================================
echo Installing Project Dependencies
echo =====================================

REM Upgrade pip first
python -m pip install --upgrade pip

REM Install dependencies from requirements.txt
python -m pip install -r requirements.txt

echo =====================================
echo Dependencies Installed Successfully!
echo =====================================

pause
