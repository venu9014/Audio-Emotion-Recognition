@echo off
set PYTHON=python
if exist .venv\Scripts\python.exe (
    set PYTHON=.venv\Scripts\python.exe
)

if "%1"=="test" (
    %PYTHON% test.py
) else if "%1"=="app" (
    if exist .venv\Scripts\streamlit.exe (
        .venv\Scripts\streamlit.exe run app.py
    ) else (
        streamlit run app.py
    )
) else if "%1"=="train" (
    %PYTHON% src\train.py --model cnn_lstm
) else if "%1"=="evaluate" (
    %PYTHON% src\evaluate.py
) else (
    echo Available commands:
    echo   make test      - Run comprehensive test suite
    echo   make app       - Run Streamlit web application
    echo   make train     - Train CNN-LSTM model
    echo   make evaluate  - Evaluate models on test set
)
