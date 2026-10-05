# Kullanım: önce `gh auth login`, sonra repo kökünde: powershell -File scripts/github_setup.ps1
# Bu dosya UTF-8 BOM ile kayıtlı: Windows PowerShell 5.1 Türkçe etiket adlarını ancak böyle doğru okur.
# Tekrar çalıştırmak güvenli: etiketler --force ile güncellenir, var olan milestone'lar atlanır.
$ErrorActionPreference = "Stop"

$labels = @(
  @("alan/python","1f6feb"), @("alan/engine","8250df"), @("alan/ui","bf3989"),
  @("alan/altyapı","6e7781"), @("alan/doküman","0e8a16"),
  @("tür/özellik","a2eeef"), @("tür/deney","fbca04"), @("tür/hata","d73a4a"),
  @("tür/spike","c5def5"), @("öncelik/yüksek","b60205")
)
foreach ($l in $labels) {
  gh label create $l[0] --color $l[1] --force
  if ($LASTEXITCODE -ne 0) { throw "Etiket oluşturulamadı: $($l[0])" }
}

$existing = @(gh api 'repos/{owner}/{repo}/milestones?state=all' --jq '.[].title')
foreach ($i in 1..10) {
  $title = "H$i"
  if ($existing -notcontains $title) {
    gh api 'repos/{owner}/{repo}/milestones' -f title=$title | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Milestone oluşturulamadı: $title" }
  }
}
Write-Host "Etiketler ve H1-H10 milestone'ları hazır."
