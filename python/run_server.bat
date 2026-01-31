@echo off
cd /d "%~dp0"
echo Installing dependencies...
pip install torch numpy pywin32 --quiet
echo.
echo Starting ML Inference Server...
echo Press Ctrl+C to stop
echo.
python ml_inference_server.py --model checkpoints/best_agent.pt -v
pause
