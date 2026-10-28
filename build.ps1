Set-Location $PSScriptRoot
pdflatex -interaction=nonstopmode main.tex | Out-Null
bibtex main | Select-String -Pattern "Warning|Error|error" 
pdflatex -interaction=nonstopmode main.tex | Out-Null
$out = pdflatex -interaction=nonstopmode main.tex
$out | Select-String -Pattern "^!|Overfull|Output written|undefined|multiply" | Select-Object -First 40
