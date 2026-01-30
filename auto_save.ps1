# Auto-save script for Star Polymarket
# Runs every 5 hours to commit and push changes to GitHub

$repoPath = "C:\Users\Star\.local\bin\star-polymarket"
$logFile = "$repoPath\auto_save.log"

# Log function
function Log($msg) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$timestamp - $msg" | Out-File -Append $logFile
    Write-Host "$timestamp - $msg"
}

Set-Location $repoPath

Log "Starting auto-save..."

# Check if there are changes
$status = git status --porcelain
if ($status) {
    Log "Changes detected, committing..."

    # Add all changes (except sensitive files)
    git add -A
    git reset -- .env *.json 2>$null

    # Commit with timestamp
    $commitMsg = "Auto-save $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
    git commit -m "$commitMsg`n`nCo-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"

    # Push to GitHub
    $pushResult = git push origin master 2>&1
    if ($LASTEXITCODE -eq 0) {
        Log "Successfully pushed to GitHub"
    } else {
        Log "Push failed: $pushResult"
    }
} else {
    Log "No changes to commit"
}

Log "Auto-save complete"
