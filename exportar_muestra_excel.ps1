param(
    [ValidateSet("train", "test")]
    [string]$Split = "train",

    [ValidateSet("head", "tail", "random", "order")]
    [string]$Mode = "random",

    [int]$Rows = 100,
    [string]$Columns = "*",
    [string]$Where = "",
    [string]$OrderBy = "",
    [switch]$Desc,
    [int]$Seed = 42,

    [ValidateSet("all", "ztautau", "ttbar", "diboson", "htautau")]
    [string]$TestSource = "all",

    [switch]$IncludeLabel,
    [switch]$IncludeWeight,
    [switch]$IncludeDer,
    [switch]$CsvOnly,
    [string]$OutputPath = "",
    [switch]$ShowSql
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-DuckDbExe {
    $duckdbCmd = Get-Command duckdb -ErrorAction SilentlyContinue
    if ($duckdbCmd) {
        return $duckdbCmd.Source
    }

    $fallback = "C:\Users\veros\AppData\Local\Microsoft\WinGet\Packages\DuckDB.cli_Microsoft.Winget.Source_8wekyb3d8bbwe\duckdb.exe"
    if (Test-Path $fallback) {
        return $fallback
    }

    throw "No se encontro duckdb.exe. Instala DuckDB CLI con: winget install --id DuckDB.cli -e"
}

function To-SqlPath([string]$path) {
    return ($path -replace "\\", "/")
}

function Get-RgbColor([int]$r, [int]$g, [int]$b) {
    return ($r + (256 * $g) + (65536 * $b))
}

function Get-FriendlyDescriptionMap {
    return @{
        "PRI_lep_pt"              = "Momento transversal del lepton detectado (energia util para separar senal y fondo)."
        "PRI_lep_eta"             = "Posicion angular eta del lepton (indica direccion respecto al haz)."
        "PRI_lep_phi"             = "Angulo phi del lepton en el plano horizontal del detector."
        "PRI_had_pt"              = "Momento transversal del tau hadronico (componente hadronica del evento)."
        "PRI_had_eta"             = "Posicion angular eta del tau hadronico."
        "PRI_had_phi"             = "Angulo phi del tau hadronico."
        "PRI_jet_leading_pt"      = "Momento del jet mas energetico (jet principal)."
        "PRI_jet_leading_eta"     = "Direccion eta del jet principal."
        "PRI_jet_leading_phi"     = "Direccion phi del jet principal."
        "PRI_jet_subleading_pt"   = "Momento del segundo jet mas energetico."
        "PRI_jet_subleading_eta"  = "Direccion eta del segundo jet."
        "PRI_jet_subleading_phi"  = "Direccion phi del segundo jet."
        "PRI_n_jets"              = "Cantidad de jets reconstruidos en el evento."
        "PRI_jet_all_pt"          = "Suma del momento transversal de todos los jets."
        "PRI_met"                 = "Energia faltante transversal (indicio de particulas no detectadas como neutrinos)."
        "PRI_met_phi"             = "Direccion angular de la energia faltante."

        "DER_mass_MMC"            = "Masa estimada del boson (aqui no disponible en bruto, se marca como -25)."
        "DER_mass_transverse_met_lep" = "Masa transversal combinando lepton y energia faltante."
        "DER_mass_vis"            = "Masa visible del sistema lepton + tau hadronico."
        "DER_pt_h"                = "Momento transversal total del candidato Higgs reconstruido."
        "DER_deltaeta_jet_jet"    = "Separacion en eta entre los dos jets principales."
        "DER_mass_jet_jet"        = "Masa invariante de los dos jets principales."
        "DER_prodeta_jet_jet"     = "Producto de eta entre jets (patron de topologia del evento)."
        "DER_deltar_tau_lep"      = "Distancia angular DeltaR entre tau hadronico y lepton."
        "DER_pt_tot"              = "Momento transversal total del evento (objetos visibles + met)."
        "DER_sum_pt"              = "Suma escalar de momentos transversales relevantes."
        "DER_pt_ratio_lep_tau"    = "Relacion de momento entre lepton y tau hadronico."
        "DER_met_phi_centrality"  = "Centralidad angular de la energia faltante respecto al sistema tau-lepton."
        "DER_lep_eta_centrality"  = "Que tan centrado esta el lepton entre los dos jets."

        "label"                   = "Etiqueta real de entrenamiento (1 = senal Higgs, 0 = fondo)."
        "weight"                  = "Peso estadistico del evento en entrenamiento."
        "weights"                 = "Peso estadistico del evento en datos de test/simulacion."
        "source_dataset"          = "Archivo/fuente de origen dentro del conjunto de test."
        "filename"                = "Ruta del archivo parquet desde donde se extrajo el evento."
    }
}

function Get-FriendlyDescription {
    param(
        [Parameter(Mandatory = $true)][string]$ColumnName,
        [Parameter(Mandatory = $true)][hashtable]$DescriptionMap
    )

    if ($DescriptionMap.ContainsKey($ColumnName)) {
        return $DescriptionMap[$ColumnName]
    }
    if ($ColumnName -like "PRI_*") {
        return "Variable primaria medida por el detector."
    }
    if ($ColumnName -like "DER_*") {
        return "Variable derivada calculada a partir de mediciones del detector."
    }
    return "Columna auxiliar del dataset."
}

function Convert-CsvToXlsx {
    param(
        [Parameter(Mandatory = $true)][string]$CsvPath,
        [Parameter(Mandatory = $true)][string]$XlsxPath,
        [string]$SheetName = "muestra"
    )

    $excel = $null
    $workbook = $null
    $worksheet = $null

    try {
        $excel = New-Object -ComObject Excel.Application
        $excel.Visible = $false
        $excel.DisplayAlerts = $false

        $workbook = $excel.Workbooks.Open($CsvPath)
        $worksheet = $workbook.Worksheets.Item(1)

        # En algunos equipos Excel abre el CSV completo en una sola columna por configuracion regional.
        # Si detectamos ese caso, lo separamos explicitamente por comas.
        $a1 = [string]$worksheet.Cells.Item(1, 1).Value2
        $b1 = [string]$worksheet.Cells.Item(1, 2).Value2
        if (($a1 -like "*,*") -and [string]::IsNullOrWhiteSpace($b1)) {
            $worksheet.Columns.Item(1).TextToColumns(
                $worksheet.Range("A1"),
                1,      # xlDelimited
                1,      # xlTextQualifierDoubleQuote
                $false, # ConsecutiveDelimiter
                $false, # Tab
                $false, # Semicolon
                $true,  # Comma
                $false, # Space
                $false, # Other
                [Type]::Missing,
                [Type]::Missing,
                ".",
                ",",
                $false
            ) | Out-Null
        }

        if ($SheetName.Length -gt 31) {
            $SheetName = $SheetName.Substring(0, 31)
        }
        $worksheet.Name = $SheetName

        $descriptionMap = Get-FriendlyDescriptionMap
        $usedBefore = $worksheet.UsedRange
        $lastCol = $usedBefore.Columns.Count
        $lastRow = $usedBefore.Rows.Count

        # Insertar fila superior: fila 1 = descripcion humana, fila 2 = nombre tecnico, fila 3+ = datos
        $worksheet.Rows("1:1").Insert(-4121) | Out-Null  # xlShiftDown

        for ($col = 1; $col -le $lastCol; $col++) {
            $officialName = [string]$worksheet.Cells.Item(2, $col).Value2
            if ([string]::IsNullOrWhiteSpace($officialName)) { continue }
            $humanText = Get-FriendlyDescription -ColumnName $officialName -DescriptionMap $descriptionMap
            $worksheet.Cells.Item(1, $col).Value2 = $humanText
        }

        $usedAfter = $worksheet.UsedRange
        $lastRowAfter = $usedAfter.Rows.Count

        $row1 = $worksheet.Range($worksheet.Cells.Item(1, 1), $worksheet.Cells.Item(1, $lastCol))
        $row2 = $worksheet.Range($worksheet.Cells.Item(2, 1), $worksheet.Cells.Item(2, $lastCol))
        $allRange = $worksheet.Range($worksheet.Cells.Item(1, 1), $worksheet.Cells.Item($lastRowAfter, $lastCol))
        $dataRange = $worksheet.Range($worksheet.Cells.Item(3, 1), $worksheet.Cells.Item($lastRowAfter, $lastCol))

        # Estilo base
        $allRange.Font.Name = "Calibri"
        $allRange.Font.Size = 10
        $allRange.VerticalAlignment = -4160  # xlTop

        # Fila 1: descripcion amigable
        $row1.Font.Bold = $true
        $row1.Font.Color = Get-RgbColor 36 52 71
        $row1.Interior.Color = Get-RgbColor 235 242 251
        $row1.WrapText = $true
        $worksheet.Rows.Item(1).RowHeight = 45

        # Fila 2: nombres oficiales
        $row2.Font.Bold = $true
        $row2.Font.Color = Get-RgbColor 255 255 255
        $row2.Interior.Color = Get-RgbColor 31 78 121
        $worksheet.Rows.Item(2).RowHeight = 24

        # Bordes suaves para ordenar lectura
        $allRange.Borders.LineStyle = 1  # xlContinuous
        $allRange.Borders.Weight = 2     # xlThin
        $allRange.Borders.Color = Get-RgbColor 220 220 220

        # Tabla elegante desde fila 2 (encabezados tecnicos + datos)
        try {
            $tableRange = $worksheet.Range($worksheet.Cells.Item(2, 1), $worksheet.Cells.Item($lastRowAfter, $lastCol))
            $listObject = $worksheet.ListObjects.Add(1, $tableRange, $null, 1)  # xlSrcRange, xlYes
            $listObject.Name = ("TablaMuestra_" + [guid]::NewGuid().ToString("N").Substring(0, 8))
            $listObject.TableStyle = "TableStyleMedium2"
        }
        catch {
            # Si no se puede crear la tabla (caso raro), al menos activamos filtro en fila 2.
            $row2.AutoFilter() | Out-Null
        }

        # Congelar panel para mantener visibles fila de descripcion y encabezado tecnico
        $worksheet.Activate() | Out-Null
        $excel.ActiveWindow.SplitColumn = 0
        $excel.ActiveWindow.SplitRow = 2
        $excel.ActiveWindow.FreezePanes = $true

        # Ajuste de columnas y altura de datos
        $worksheet.Columns.AutoFit() | Out-Null
        for ($col = 1; $col -le $lastCol; $col++) {
            $w = [double]$worksheet.Columns.Item($col).ColumnWidth
            if ($w -gt 45) { $worksheet.Columns.Item($col).ColumnWidth = 45 }
            if ($w -lt 10) { $worksheet.Columns.Item($col).ColumnWidth = 10 }
        }
        $dataRange.WrapText = $false

        $xlOpenXmlWorkbook = 51
        $workbook.SaveAs($XlsxPath, $xlOpenXmlWorkbook)
        $workbook.Close($false)
        $excel.Quit()
    }
    finally {
        if ($worksheet) { [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($worksheet) }
        if ($workbook) { [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($workbook) }
        if ($excel) { [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($excel) }
        [GC]::Collect()
        [GC]::WaitForPendingFinalizers()
    }
}

if ($Rows -le 0) {
    throw "Rows debe ser mayor que 0."
}

$excelMaxRows = 1048576
if ($Rows -gt $excelMaxRows) {
    Write-Warning "Excel soporta maximo $excelMaxRows filas por hoja. Se usara ese maximo."
    $Rows = $excelMaxRows
}

$duckdbExe = Resolve-DuckDbExe
$projectRoot = Split-Path -Parent $PSScriptRoot
$inputRoot = Join-Path $projectRoot "01_Datos\Competencia_CERN\HiggsML_Uncertainty_Challenge(2025)\public_data_CERN\input_data"

$outputDir = Join-Path $projectRoot "04_Resultados\Extractos_Excel"
New-Item -Path $outputDir -ItemType Directory -Force | Out-Null

if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $ext = if ($CsvOnly) { "csv" } else { "xlsx" }
    $OutputPath = Join-Path $outputDir ("muestra_{0}_{1}_{2}_{3}.{4}" -f $Split, $Mode, $Rows, $stamp, $ext)
}

$outputAbs = [System.IO.Path]::GetFullPath($OutputPath)

$whereClause = ""
if (-not [string]::IsNullOrWhiteSpace($Where)) {
    $whereClause = "WHERE ($Where)"
}

if ($Split -eq "train") {
    $featuresPath = To-SqlPath (Join-Path $inputRoot "train\data\data.parquet")
    $labelsPath = To-SqlPath (Join-Path $inputRoot "train\labels\data.labels")
    $weightsPath = To-SqlPath (Join-Path $inputRoot "train\weights\data.weights")

    if ($IncludeLabel -and $IncludeWeight) {
        $baseCte = @"
WITH
base AS (
    SELECT
        d.*,
        CAST(l.column0 AS INTEGER) AS label,
        CAST(w.column0 AS DOUBLE) AS weight
    FROM read_parquet('$featuresPath') d
    POSITIONAL JOIN read_csv_auto('$labelsPath', header=false) l
    POSITIONAL JOIN read_csv_auto('$weightsPath', header=false) w
)
"@
    }
    elseif ($IncludeLabel) {
        $baseCte = @"
WITH
base AS (
    SELECT
        d.*,
        CAST(l.column0 AS INTEGER) AS label
    FROM read_parquet('$featuresPath') d
    POSITIONAL JOIN read_csv_auto('$labelsPath', header=false) l
)
"@
    }
    elseif ($IncludeWeight) {
        $baseCte = @"
WITH
base AS (
    SELECT
        d.*,
        CAST(w.column0 AS DOUBLE) AS weight
    FROM read_parquet('$featuresPath') d
    POSITIONAL JOIN read_csv_auto('$weightsPath', header=false) w
)
"@
    }
    else {
        $baseCte = @"
WITH
base AS (
    SELECT *
    FROM read_parquet('$featuresPath')
)
"@
    }
}
else {
    $testFileMap = @{
        "ztautau" = "ztautau_data.parquet"
        "ttbar"   = "ttbar_data.parquet"
        "diboson" = "diboson_data.parquet"
        "htautau" = "htautau_data.parquet"
    }

    if ($TestSource -eq "all") {
        $testPath = To-SqlPath (Join-Path $inputRoot "test\data\*.parquet")
    }
    else {
        $testPath = To-SqlPath (Join-Path $inputRoot ("test\data\" + $testFileMap[$TestSource]))
    }

    $baseCte = @"
WITH
base AS (
    SELECT
        *,
        parse_filename(filename, true) AS source_dataset
    FROM read_parquet('$testPath', filename=true)
)
"@
}

$sourceTable = "base"
if ($IncludeDer) {
    $baseCte += @"
,
base_with_der AS (
    SELECT
        b.*,
        CAST(-25.0 AS DOUBLE) AS DER_mass_MMC,
        sqrt(greatest(
            0.0,
            2.0 * b.PRI_lep_pt * b.PRI_met *
            (1.0 - cos(atan2(sin(b.PRI_lep_phi - b.PRI_met_phi), cos(b.PRI_lep_phi - b.PRI_met_phi))))
        )) AS DER_mass_transverse_met_lep,
        sqrt(greatest(
            0.0,
            2.0 * b.PRI_had_pt * b.PRI_lep_pt *
            (
                cosh(b.PRI_had_eta - b.PRI_lep_eta) -
                cos(atan2(sin(b.PRI_had_phi - b.PRI_lep_phi), cos(b.PRI_had_phi - b.PRI_lep_phi)))
            )
        )) AS DER_mass_vis,
        sqrt(
            power(
                b.PRI_had_pt * cos(b.PRI_had_phi) +
                b.PRI_lep_pt * cos(b.PRI_lep_phi) +
                b.PRI_met * cos(b.PRI_met_phi),
                2
            ) +
            power(
                b.PRI_had_pt * sin(b.PRI_had_phi) +
                b.PRI_lep_pt * sin(b.PRI_lep_phi) +
                b.PRI_met * sin(b.PRI_met_phi),
                2
            )
        ) AS DER_pt_h,
        CASE
            WHEN b.PRI_n_jets >= 2 AND b.PRI_jet_leading_pt > -20 AND b.PRI_jet_subleading_pt > -20
            THEN abs(b.PRI_jet_leading_eta - b.PRI_jet_subleading_eta)
            ELSE -25.0
        END AS DER_deltaeta_jet_jet,
        CASE
            WHEN b.PRI_n_jets >= 2 AND b.PRI_jet_leading_pt > -20 AND b.PRI_jet_subleading_pt > -20
            THEN sqrt(greatest(
                0.0,
                2.0 * b.PRI_jet_leading_pt * b.PRI_jet_subleading_pt *
                (
                    cosh(b.PRI_jet_leading_eta - b.PRI_jet_subleading_eta) -
                    cos(atan2(
                        sin(b.PRI_jet_leading_phi - b.PRI_jet_subleading_phi),
                        cos(b.PRI_jet_leading_phi - b.PRI_jet_subleading_phi)
                    ))
                )
            ))
            ELSE -25.0
        END AS DER_mass_jet_jet,
        CASE
            WHEN b.PRI_n_jets >= 2 AND b.PRI_jet_leading_pt > -20 AND b.PRI_jet_subleading_pt > -20
            THEN b.PRI_jet_leading_eta * b.PRI_jet_subleading_eta
            ELSE -25.0
        END AS DER_prodeta_jet_jet,
        sqrt(
            power(b.PRI_had_eta - b.PRI_lep_eta, 2) +
            power(atan2(sin(b.PRI_had_phi - b.PRI_lep_phi), cos(b.PRI_had_phi - b.PRI_lep_phi)), 2)
        ) AS DER_deltar_tau_lep,
        sqrt(
            power(
                b.PRI_had_pt * cos(b.PRI_had_phi) +
                b.PRI_lep_pt * cos(b.PRI_lep_phi) +
                b.PRI_met * cos(b.PRI_met_phi) +
                CASE WHEN b.PRI_n_jets >= 1 AND b.PRI_jet_leading_pt > -20
                     THEN b.PRI_jet_leading_pt * cos(b.PRI_jet_leading_phi) ELSE 0.0 END +
                CASE WHEN b.PRI_n_jets >= 2 AND b.PRI_jet_subleading_pt > -20
                     THEN b.PRI_jet_subleading_pt * cos(b.PRI_jet_subleading_phi) ELSE 0.0 END,
                2
            ) +
            power(
                b.PRI_had_pt * sin(b.PRI_had_phi) +
                b.PRI_lep_pt * sin(b.PRI_lep_phi) +
                b.PRI_met * sin(b.PRI_met_phi) +
                CASE WHEN b.PRI_n_jets >= 1 AND b.PRI_jet_leading_pt > -20
                     THEN b.PRI_jet_leading_pt * sin(b.PRI_jet_leading_phi) ELSE 0.0 END +
                CASE WHEN b.PRI_n_jets >= 2 AND b.PRI_jet_subleading_pt > -20
                     THEN b.PRI_jet_subleading_pt * sin(b.PRI_jet_subleading_phi) ELSE 0.0 END,
                2
            )
        ) AS DER_pt_tot,
        b.PRI_had_pt + b.PRI_lep_pt +
            CASE WHEN b.PRI_jet_all_pt > -20 THEN b.PRI_jet_all_pt ELSE 0.0 END AS DER_sum_pt,
        b.PRI_lep_pt / NULLIF(b.PRI_had_pt, 0.0) AS DER_pt_ratio_lep_tau,
        CASE
            WHEN abs(sin(b.PRI_had_phi - b.PRI_lep_phi)) < 1e-9 THEN -25.0
            ELSE
                (
                    (sin(b.PRI_met_phi - b.PRI_lep_phi) / sin(b.PRI_had_phi - b.PRI_lep_phi)) +
                    (sin(b.PRI_had_phi - b.PRI_met_phi) / sin(b.PRI_had_phi - b.PRI_lep_phi))
                ) / NULLIF(
                    sqrt(
                        power(sin(b.PRI_met_phi - b.PRI_lep_phi) / sin(b.PRI_had_phi - b.PRI_lep_phi), 2) +
                        power(sin(b.PRI_had_phi - b.PRI_met_phi) / sin(b.PRI_had_phi - b.PRI_lep_phi), 2)
                    ),
                    0.0
                )
        END AS DER_met_phi_centrality,
        CASE
            WHEN b.PRI_n_jets >= 2
             AND b.PRI_jet_leading_pt > -20
             AND b.PRI_jet_subleading_pt > -20
             AND abs(b.PRI_jet_leading_eta - b.PRI_jet_subleading_eta) > 1e-9
            THEN exp(
                -4.0 * power(
                    b.PRI_lep_eta - ((b.PRI_jet_leading_eta + b.PRI_jet_subleading_eta) / 2.0),
                    2
                ) / power(b.PRI_jet_leading_eta - b.PRI_jet_subleading_eta, 2)
            )
            ELSE -25.0
        END AS DER_lep_eta_centrality
    FROM base b
)
"@
    $sourceTable = "base_with_der"
}

$orderDirection = if ($Desc.IsPresent) { "DESC" } else { "ASC" }

switch ($Mode) {
    "head" {
        $finalQuery = @"
$baseCte
SELECT $Columns
FROM $sourceTable
$whereClause
LIMIT $Rows
"@
    }
    "tail" {
        $tailColumns = $Columns
        if ($Columns.Trim() -eq "*") {
            $tailColumns = "* EXCLUDE (_row_id)"
        }
        $finalQuery = @"
$baseCte
SELECT $tailColumns
FROM (
    SELECT row_number() OVER () AS _row_id, *
    FROM (
        SELECT *
        FROM $sourceTable
        $whereClause
    ) b
) q
ORDER BY _row_id DESC
LIMIT $Rows
"@
    }
    "order" {
        if ([string]::IsNullOrWhiteSpace($OrderBy)) {
            throw "Para Mode=order debes indicar -OrderBy (ej: -OrderBy ""PRI_met"")."
        }
        $finalQuery = @"
$baseCte
SELECT $Columns
FROM $sourceTable
$whereClause
ORDER BY $OrderBy $orderDirection
LIMIT $Rows
"@
    }
    "random" {
        $finalQuery = @"
$baseCte
SELECT $Columns
FROM (
    SELECT *
    FROM $sourceTable
    $whereClause
) q
USING SAMPLE $Rows ROWS (reservoir, $Seed)
"@
    }
    default {
        throw "Modo no soportado: $Mode"
    }
}

if ($ShowSql) {
    Write-Host "----- SQL generado -----" -ForegroundColor Cyan
    Write-Host $finalQuery
    Write-Host "------------------------" -ForegroundColor Cyan
}

$tempCsv = Join-Path $env:TEMP ("higgs_extract_{0}.csv" -f ([guid]::NewGuid().ToString("N")))

try {
    Write-Host "Ejecutando consulta en DuckDB..." -ForegroundColor Yellow
    & $duckdbExe -csv -header -c $finalQuery | Set-Content -Path $tempCsv -Encoding UTF8

    if (-not (Test-Path $tempCsv)) {
        throw "No se pudo crear el CSV temporal."
    }

    if ($CsvOnly) {
        Copy-Item -Path $tempCsv -Destination $outputAbs -Force
        Write-Host "Archivo CSV exportado en: $outputAbs" -ForegroundColor Green
        return
    }

    Write-Host "Convirtiendo CSV a Excel..." -ForegroundColor Yellow
    Convert-CsvToXlsx -CsvPath ([System.IO.Path]::GetFullPath($tempCsv)) -XlsxPath $outputAbs -SheetName ("{0}_{1}" -f $Split, $Mode)
    Write-Host "Archivo Excel exportado en: $outputAbs" -ForegroundColor Green
}
finally {
    if (Test-Path $tempCsv) {
        Remove-Item -Path $tempCsv -Force
    }
}
