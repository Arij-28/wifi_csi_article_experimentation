$seeds = @(42, 123, 999)
$users = @(1, 3, 5)

Write-Host "=== ERM ==="
foreach ($u in $users) {
  foreach ($s in $seeds) {
    python .\scripts\train_widar_selective_scalable.py `
      --subset_train 12000 `
      --subset_test -1 `
      --lambda_proto 0.0 `
      --lambda_brier 0.0 `
      --user_test $u `
      --epochs 15 `
      --seed $s `
      --tag erm_large
  }
}

Write-Host "=== ERM + BRIER ==="
foreach ($u in $users) {
  foreach ($s in $seeds) {
    python .\scripts\train_widar_selective_scalable.py `
      --subset_train 12000 `
      --subset_test -1 `
      --lambda_proto 0.0 `
      --lambda_brier 0.1 `
      --user_test $u `
      --epochs 15 `
      --seed $s `
      --tag erm_brier_large
  }
}

Write-Host "=== ERM + PROTO ==="
foreach ($u in $users) {
  foreach ($s in $seeds) {
    python .\scripts\train_widar_selective_scalable.py `
      --subset_train 12000 `
      --subset_test -1 `
      --lambda_proto 0.003 `
      --lambda_brier 0.0 `
      --user_test $u `
      --epochs 15 `
      --seed $s `
      --tag erm_proto_large
  }
}