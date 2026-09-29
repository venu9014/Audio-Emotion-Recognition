param (
    [string]$Target = "help"
)

$PythonExe = "python"
if (Test-Path ".\.venv\Scripts\python.exe") {
    $PythonExe = ".\.venv\Scripts\python.exe"
}

switch ($Target.ToLower()) {
    "test" {
        & $PythonExe test.py
    }
    "app" {
        if (Test-Path ".\.venv\Scripts\streamlit.exe") {
            & ".\.venv\Scripts\streamlit.exe" run app.py
        } else {
            streamlit run app.py
        }
    }
    "train" {
        & $PythonExe src/train.py --model cnn_lstm
    }
    "evaluate" {
        & $PythonExe src/evaluate.py
    }
    default {
        Write-Host "Available commands:" -ForegroundColor Cyan
        Write-Host "  .\make test      - Run comprehensive test suite"
        Write-Host "  .\make app       - Run Streamlit web application"
        Write-Host "  .\make train     - Train CNN-LSTM model"
        Write-Host "  .\make evaluate  - Evaluate models on test set"
    }
}
