for TRIAL in {1..20}
do
  python3 search/run_search.py -w $TRIAL -c search/config/experiment/experiment_small.tml & 
done
