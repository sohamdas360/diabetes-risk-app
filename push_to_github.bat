@echo off
echo ===================================
echo   Pushing Code to GitHub
echo ===================================
echo.

cd "c:\Users\ACER\Desktop\ML 2\Prediction"

echo [1/3] Staging changes...
git add .

echo [2/3] Committing changes...
git commit -m "ULTIMATE FIX: XGBoost feature names error (numpy array + validation bypass)"

echo [3/3] Pushing to GitHub...
git push origin master

echo.
echo ===================================
echo   DONE! Check GitHub now.
echo ===================================
pause
