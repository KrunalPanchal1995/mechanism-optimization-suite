# Offline (reads samplefile.txt → saves convergence_plot.png)
python3.9 /home/krithika/mechanism-optimization-suite/src/convergence_plotter.py

# Live, per-generation update
python /home/krithika/mechanism-optimization-suite/src/convergence_plotter.py --live

# Live, every 100 function calls from Objective.txt
python /home/krithika/mechanism-optimization-suite/src/convergence_plotter.py --live --every-n 100

# Custom paths + slower polling + custom title
python /home/krithika/mechanism-optimization-suite/src/convergence_plotter.py --live \
    --samplefile run2/samplefile.txt \
    --outfile    run2_conv.png \
    --interval   5 \
    --title "n-heptane HTC stage"
