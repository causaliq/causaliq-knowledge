# generated from causaliq@c7917d9e7fa8466b07548514cdf37d494b861321 src/causaliq/devtools/templates.py - do not edit
param(
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Args
)

& python "$PSScriptRoot\check_ci.py" @Args
exit $LASTEXITCODE
