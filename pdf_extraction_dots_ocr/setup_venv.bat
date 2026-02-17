@echo off
REM Setup script for DOTS OCR venv_dot_ocr on Windows

echo Creating virtual environment: venv_dot_ocr
python -m venv venv_dot_ocr

echo.
echo Activating virtual environment...
call venv_dot_ocr\Scripts\activate.bat

echo.
echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Installing requirements...
pip install -r requirements.txt

echo.
echo Setup complete!
echo.
echo To activate this environment in the future:
echo   venv_dot_ocr\Scripts\activate
echo.
echo To run the Gradio app:
echo   python gradio_app.py
echo.
pause
