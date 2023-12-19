function="real_functions"
A_TOTAL="79"

log_path="${function}_logs"

if [ ! -d "$log_path" ]; then
  mkdir -p "$log_path"
fi

for noise in $(seq 0 0.01 0.05 0.1 0.2); do
  for B in {0..6}; do
      UUID="job_$(uuidgen | cut -c 1-8)"
      log_name="${log_path}/logs_${UUID}.out"
      qsub -o ${log_name} -e ${log_name} -N ${UUID} -J 0-${A_TOTAL} -v B=$B,function=$function, noise=$noise scripts/array_job.sh
  done
done
