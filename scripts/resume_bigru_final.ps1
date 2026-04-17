$seeds = @(42, 123, 999)
$users = @(1, 3, 5)

function Run-IfMissing {
    param(
        [string]$Tag,
        [int]$User,
        [int]$Seed,
        [double]$LambdaProto
    )

    $lp = $LambdaProto.ToString("0.000").Replace(".", "_")

    if ($LambdaProto -eq 0.0) {
        $lp = "0_0"
    }

    $summary = ".\outputs\${Tag}_user${User}_train-1_test-1_lp${lp}_seed${Seed}_summary.json"

    if (Test-Path $summary) {
        Write-Host "[SKIP] $summary"
        return
    }

    Write-Host "[RUN ] tag=$Tag user=$User seed=$Seed lambda_proto=$LambdaProto"

    python .\scripts\train_widar_bigru_proto_selective.py `
        --subset_train -1 `
        --subset_test -1 `
        --lambda_proto $LambdaProto `
        --user_test $User `
        --epochs 20 `
        --seed $Seed `
        --tag $Tag
}

Write-Host "=== BiGRU ERM ==="
foreach ($u in $users) {
    foreach ($s in $seeds) {
        Run-IfMissing -Tag "bigru_erm" -User $u -Seed $s -LambdaProto 0.0
    }
}

Write-Host "=== BiGRU + Proto(0.001) ==="
foreach ($u in $users) {
    foreach ($s in $seeds) {
        Run-IfMissing -Tag "bigru_proto001" -User $u -Seed $s -LambdaProto 0.001
    }
}