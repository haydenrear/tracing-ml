# shellcheck disable=SC2160
while [ true ]
do
  rm loss_file.txt
  rm out_file.txt
  rm q.txt
  found=$(python3 test/tracing_ml/test_q_deformed_softmax.py || true)
  echo "${found} was found!"
  echo "Restarting again!"
done