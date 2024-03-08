run=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --nsamples)
            nsamples=$2
            shift 2
            ;;
        --batch_size)
            batch_size=$2
            shift 2
            ;;
        --run)
            run=true
            shift
            ;;
        --mode)
            mode=$2
            shift 2
            ;;
            *)
        echo "Unknown parameter: $1"
        exit 1
        ;;
    esac
done

if [[ -z "$nsamples" ]]; then
    echo "nsamples is required"
    exit 1
fi

if [[ -z "$batch_size" ]]; then
    echo "batch_size is required"
    exit 1
fi

if [[ -z "$mode" ]]; then
    echo "mode is required"
    exit 1
fi

dir=<your dir here>

if [ $(($nsamples % $batch_size)) -ne 0 ]; then
    echo "Error: nsamples is not divisible by batch_size" >&2
    exit 1
fi
nbatch=$(( nsamples / batch_size ))

declare -A offsets=( ["train"]=10000 ["valid"]=20000 ["test"]=30000)
offset=${offsets[$mode]}

for batch in $(seq 1 $nbatch); do
    printf "\n\n\n\nBatch ${batch}\n"
    date
    key=$(( offset + batch ))
    cmd="python CFD_multi_Hydra.py +args=2D_Multi_Rand ++args.init_key=${key} ++args.numbers=${batch_size} ++args.mode=${mode} ++args.save=${dir}";
    echo $cmd
    if $run; then 
        eval $cmd
    fi
done

date
