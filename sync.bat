@echo off
echo ===================================
echo Syncing project to GitHub...
echo ===================================

git add .

set "msg="
set /p msg="Enter commit message (press Enter for default 'Automated sync'): "
if not defined msg set "msg=Automated sync"

git commit -m "%msg%"
git push

echo.
echo Sync complete!
pause
