UUID = $(uuidgen)
function = $(rkhs_functions)
A_TOTAL = $(200)

for i in {0..5}; do
    qsub -v UUID=$UUID B=$i A_TOTAL=$A_TOTAL function=$function run.sh
