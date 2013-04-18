#!/bin/bash


while getopts ":f:" opt; do
	case $opt in
		f)
			INP=$OPTARG
      		;;
    	\?)
      		echo "Invalid option: -$OPTARG" >&2
      		exit 1
      		;;
    	:)
      		echo "Option -$OPTARG requires an argument." >&2
			exit 1
			;;
	esac
done

INFILE="./data/$INP"
SIMFILE="./sims/$INP.sim"
echo -e "Simulation Starting:\n\n" >$SIMFILE

for proc in 48 40 32 24 16 8 4 2 1
do
	MFILE="./models/$INP$proc.model"
	echo -e "\nProcessors: $proc:\n" >>$SIMFILE
	{ time mpiexec -n $proc -l ./parallel -f $INFILE -m $MFILE -l ; } 2>>$SIMFILE
	sleep 3
done

