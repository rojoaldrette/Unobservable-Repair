# Exporta el póster a PNG con PowerPoint (para revisión visual).
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$pptx = Join-Path $here "poster_tesis.pptx"
$png = Join-Path $here "poster_tesis.png"
$pp = New-Object -ComObject PowerPoint.Application
$pres = $pp.Presentations.Open($pptx, $true, $false, $false)
$pres.Slides(1).Export($png, "PNG", 2700, 3600)
$pres.Close()
$pp.Quit()
Write-Output "png: $png"
