#!/usr/bin/bash
NUM_OMP=16          # Number of OpenMP parallelization
NUM_MPI=1           # Number of MPI parallelization
INPUT=.py    # Input file (namelist)
OUTPUT=out.log      # Standard output file
STATUS=job.status

export OMP_NUM_THREADS=${NUM_OMP}
echo ${NUM_OMP} threads OpenMP parallelization 

if [ $(( NUM_MPI )) -eq 1 ]; then
    echo No MPI parallelization
    nohup smilei ${INPUT} > ${OUTPUT} & 
else
    echo ${NUM_MPI} threads MPI parallelization
    nohup mpirun -np ${NUM_MPI} smilei ${INPUT} > ${OUTPUT} & 
fi

JOBID=$!
START=`LANG='C' date`
echo PID: ${JOBID}

# write the job status to $STATUS file
echo ---------------------------- > ${STATUS}
echo Smilei PIC code, job status  >> ${STATUS}
echo ---------------------------- >> ${STATUS}
echo ${START} >> ${STATUS}
echo >> ${STATUS}
echo -e 'PID:\t\t'${JOBID} >> ${STATUS}
echo -e 'Input:\t\t'${INPUT} >> ${STATUS}
echo -e 'OpenMP:\t\t'${NUM_OMP} threads >> ${STATUS}
echo -e 'MPI:\t\t'${NUM_MPI} threads >> ${STATUS}

echo Job status is written in \"${STATUS}\"

