# Ежедневный пересчёт плана закупок.
#
# Регистрация в Планировщике Windows (один раз, из PowerShell; путь к проекту поправить):
#   schtasks /Create /TN "Планирование закупок" /SC DAILY /ST 07:00 `
#     /TR "powershell -NoProfile -ExecutionPolicy Bypass -File C:\путь\к\проекту\scripts\run_daily.ps1"
# Проверить запуск вручную:  schtasks /Run /TN "Планирование закупок"
# Убрать задачу:             schtasks /Delete /TN "Планирование закупок" /F
#
# Дата расчёта («сегодня»): параметр -Today ГГГГ-ММ-ДД или переменная окружения DATA_DATE,
# например  run_daily.ps1 -Today 2026-08-13. Она передаётся в --today. Без неё дата берётся
# из последнего заказа/прихода в выгрузке, и в журнал пишется предупреждение.
# Для Планировщика дату удобно задать один раз:  setx DATA_DATE 2026-08-13
# и обновлять вместе с выгрузкой.
#
# Что делает: переходит в корень проекта, берёт свежую выгрузку из data/raw/share
# и обновляет out/ (дашборд, plan.csv, risks.csv, summary.md) и снимок в out/history/.
# Журнал запуска пишется в out/run_daily.log.
#
# Ошибки python (в том числе DataError) идут в stderr. В PowerShell 5.1 при
# $ErrorActionPreference = "Stop" такая строка через 2>&1 прерывает скрипт,
# поэтому на время запуска python ставится "Continue", а код выхода пишется в журнал.

param(
    [string]$Today = $env:DATA_DATE
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
New-Item -ItemType Directory -Force out | Out-Null

$log = "out/run_daily.log"
$env:PYTHONIOENCODING = "utf-8"
"=== $(Get-Date -Format 'dd.MM.yyyy HH:mm') ===" | Out-File -FilePath $log -Append -Encoding utf8

$runArgs = @("-m", "zakupki", "run", "--data", "data/raw/share")
if ($Today) {
    $runArgs += @("--today", $Today)
}

$ErrorActionPreference = "Continue"
$code = 1
try {
    python @runArgs 2>&1 |
        ForEach-Object { "$_" } |
        Out-File -FilePath $log -Append -Encoding utf8
    $code = $LASTEXITCODE
} catch {
    "Сбой запуска: $($_.Exception.Message)" | Out-File -FilePath $log -Append -Encoding utf8
}
"Код выхода: $code" | Out-File -FilePath $log -Append -Encoding utf8
exit $code
