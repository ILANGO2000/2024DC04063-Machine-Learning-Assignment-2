@echo off
echo ==========================================
echo  RecoMart Pipeline Dependency Installer
echo ==========================================

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in your PATH.
    echo Please install Python 3.10+ from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [INFO] Python found. Installing dependencies...
echo.

:: Install dependencies
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo [SUCCESS] All dependencies installed successfully!
echo You can now run the pipeline scripts.
echo.
pause
