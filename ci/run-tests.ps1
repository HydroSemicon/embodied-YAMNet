$ErrorActionPreference = "Stop"
$env:PYTHONPATH = (Resolve-Path "$PSScriptRoot\..\src").Path
python -m unittest discover -s "$PSScriptRoot\..\tests" -v
