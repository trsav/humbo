function="rkhs_functions"
A_TOTAL="200"

for B in {0..5}; do
    UUID="job_$(uuidgen | cut -c 1-8)"
    qsub -N ${UUID} -J 1-${A_TOTAL} -v B=$B,function=$function scripts/array_job.sh
done