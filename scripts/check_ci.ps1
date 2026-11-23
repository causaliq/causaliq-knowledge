# generated from causaliq@27a2f560f378 src/causaliq/devtools/templates.py - do not edit
param(
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Args
)

& python "$PSScriptRoot\check_ci.py" @Args
exit $LASTEXITCODE
