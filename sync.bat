@echo off
echo ===================================
echo Syncing project to GitHub...
echo ===================================

git add .

set /p msg="Enter commit message (press Enter for default 'Automated sync'): "
if "%msg%"=="" set msg="Automated sync"

git commit -m "%msg%"
git push

echo.
echo Sync complete!
pause
