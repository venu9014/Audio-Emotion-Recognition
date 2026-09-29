.PHONY: test app train evaluate help

PYTHON = python

help:
	@echo Available commands:
	@echo   make test      - Run comprehensive test suite
	@echo   make app       - Run Streamlit web application
	@echo   make train     - Train CNN-LSTM model
	@echo   make evaluate  - Evaluate models on test set

test:
	$(PYTHON) test.py

app:
	streamlit run app.py

train:
	$(PYTHON) src/train.py --model cnn_lstm

evaluate:
	$(PYTHON) src/evaluate.py
